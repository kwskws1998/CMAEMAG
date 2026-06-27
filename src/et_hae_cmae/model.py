import torch

from src.configs.constants import TaskTypes
from src.configs.data import DataArgs
from src.configs.trainers import TrainerDL
from src.et_hae_cmae.conditional_mae import ConditionalGazeMAE
from src.et_hae_cmae.config import CMAEMAGEye
from src.models.base_model import register_model
from src.models.mag_model import MAGModel


@register_model
class CMAEMAGEyeModel(MAGModel):
    """MAG baseline with a conditional masked gaze autoencoder."""

    def __init__(
        self,
        model_args: CMAEMAGEye,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ) -> None:
        super().__init__(
            model_args=model_args,
            trainer_args=trainer_args,
            data_args=data_args,
        )
        self.model_args = model_args
        self.cmae_loss_weight = model_args.cmae_loss_weight
        self.cmae_detach_text_condition = model_args.cmae_detach_text_condition

        self.cmae = ConditionalGazeMAE(
            gaze_dim=model_args.eyes_dim,
            text_dim=model_args.text_dim,
            hidden_dim=model_args.cmae_hidden_dim,
            decoder_dim=model_args.cmae_decoder_dim,
            num_encoder_layers=model_args.cmae_num_encoder_layers,
            num_decoder_layers=model_args.cmae_num_decoder_layers,
            num_attention_heads=model_args.cmae_num_attention_heads,
            dropout=model_args.cmae_dropout,
            mask_ratio=model_args.cmae_mask_ratio,
            residual_scale=model_args.cmae_residual_scale,
        )
        self.save_hyperparameters()

    def shared_step(
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)

        assert batch_data.input_masks is not None, 'input_masks cannot be None'
        assert batch_data.input_ids is not None, 'input_ids cannot be None'
        assert batch_data.grouped_inversions is not None, (
            'grouped_inversions cannot be None'
        )

        gaze_features, gaze_positions, attention_mask = self.finalize_eye_data(
            batch_data
        )
        assert gaze_features is not None, 'gaze_features cannot be None'

        labels = batch_data.labels
        if self.task == TaskTypes.REGRESSION:
            labels = labels.squeeze().float()

        text_condition = self._token_text_condition(
            input_ids=batch_data.input_ids,
            attention_mask=attention_mask,
        )

        cmae_output = self.cmae(
            gaze=gaze_features,
            text_condition=text_condition,
            attention_mask=attention_mask,
        )

        output = self(
            input_ids=batch_data.input_ids,
            attention_mask=attention_mask,
            labels=labels,
            gaze_features=cmae_output.gaze_for_prediction,
            gaze_positions=gaze_positions,
            output_hidden_states=True,
        )

        logits = output.logits
        if self.task == TaskTypes.REGRESSION:
            logits = logits.squeeze()

        supervised_loss = self.loss(logits, labels)
        loss = supervised_loss + (
            self.cmae_loss_weight * cmae_output.reconstruction_loss
        )
        self._log_cmae_losses(
            supervised_loss=supervised_loss,
            reconstruction_loss=cmae_output.reconstruction_loss,
        )

        return labels, loss, logits

    def _token_text_condition(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedding_module = self.model.roberta.embeddings
        text_condition = embedding_module(input_ids=input_ids)
        if self.cmae_detach_text_condition:
            text_condition = text_condition.detach()
        mask = attention_mask.unsqueeze(-1).to(dtype=text_condition.dtype)
        return text_condition * mask

    def _log_cmae_losses(
        self,
        supervised_loss: torch.Tensor,
        reconstruction_loss: torch.Tensor,
    ) -> None:
        log_context = self._cmae_log_context()
        if log_context is None:
            return

        stage, add_dataloader_idx = log_context
        self.log(
            name=f'loss/supervised_{stage}',
            value=supervised_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
            add_dataloader_idx=add_dataloader_idx,
            sync_dist=True,
        )
        self.log(
            name=f'loss/cmae_reconstruction_{stage}',
            value=reconstruction_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=self.batch_size,
            add_dataloader_idx=add_dataloader_idx,
            sync_dist=True,
        )

    def _cmae_log_context(self) -> tuple[str, bool] | None:
        if self.training:
            return 'train', False

        try:
            trainer = self.trainer
        except RuntimeError:
            return None

        if trainer.validating:
            return 'val', True
        if trainer.testing:
            return 'test', True
        return None
