from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ConditionalMAEOutput:
    gaze_for_prediction: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_loss: torch.Tensor
    mask: torch.Tensor


class ConditionalGazeMAE(nn.Module):
    """Conditional MAE for token-aligned EyeBench gaze features."""

    def __init__(
        self,
        gaze_dim: int,
        text_dim: int,
        hidden_dim: int,
        decoder_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_attention_heads: int,
        dropout: float,
        mask_ratio: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.residual_scale = residual_scale

        self.gaze_projection = nn.Linear(gaze_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.decoder_projection = nn.Linear(hidden_dim, decoder_dim)
        self.decoder_text_projection = nn.Linear(text_dim, decoder_dim)
        self.decoder_mask_token = nn.Parameter(torch.zeros(decoder_dim))
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_attention_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.reconstruction_head = nn.Linear(decoder_dim, gaze_dim)
        self.refinement_head = nn.Linear(hidden_dim, gaze_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.decoder_mask_token, std=0.02)

    def forward(
        self,
        gaze: torch.Tensor,
        text_condition: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> ConditionalMAEOutput:
        valid_tokens = self._valid_gaze_tokens(gaze, attention_mask)
        mask = self._sample_mask(valid_tokens)

        hidden = self.gaze_projection(gaze) + self.text_projection(text_condition)
        hidden = self.input_norm(hidden)
        key_padding_mask = None if attention_mask is None else ~attention_mask.bool()
        encoded = self._encode_visible(
            hidden=hidden,
            mask=mask,
            attention_mask=attention_mask,
        )

        decoder_hidden = self.decoder_projection(encoded) + self.decoder_text_projection(
            text_condition
        )
        decoder_mask_token = self.decoder_mask_token.view(1, 1, -1).to(
            dtype=decoder_hidden.dtype,
            device=decoder_hidden.device,
        )
        decoder_hidden = torch.where(
            mask.unsqueeze(-1),
            decoder_hidden + decoder_mask_token,
            decoder_hidden,
        )
        decoder_hidden = self.decoder_norm(decoder_hidden)
        decoder_hidden = self.decoder(
            decoder_hidden,
            src_key_padding_mask=key_padding_mask,
        )
        reconstruction = self.reconstruction_head(decoder_hidden)

        reconstruction_loss = self._masked_mse(
            reconstruction=reconstruction,
            target=gaze,
            mask=mask,
        )

        refinement = self.refinement_head(self.dropout(encoded))
        refined_gaze = gaze + self.residual_scale * refinement
        refined_gaze = torch.where(mask.unsqueeze(-1), reconstruction, refined_gaze)
        refined_gaze = torch.where(valid_tokens.unsqueeze(-1), refined_gaze, gaze)

        return ConditionalMAEOutput(
            gaze_for_prediction=refined_gaze,
            reconstruction=reconstruction,
            reconstruction_loss=reconstruction_loss,
            mask=mask,
        )

    def _valid_gaze_tokens(
        self,
        gaze: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        nonzero_gaze = gaze.abs().sum(dim=-1) > 0
        if attention_mask is None:
            return nonzero_gaze
        return nonzero_gaze & attention_mask.bool()

    def _sample_mask(self, valid_tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_ratio == 0.0:
            return torch.zeros_like(valid_tokens)

        random_values = torch.rand(valid_tokens.shape, device=valid_tokens.device)
        mask = (random_values < self.mask_ratio) & valid_tokens

        needs_mask = valid_tokens.any(dim=1) & ~mask.any(dim=1)
        if needs_mask.any():
            valid_scores = random_values.masked_fill(~valid_tokens, 2.0)
            forced_positions = valid_scores.argmin(dim=1)
            row_ids = torch.arange(mask.shape[0], device=mask.device)
            mask[row_ids[needs_mask], forced_positions[needs_mask]] = True

        return mask

    def _encode_visible(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            visible_tokens = ~mask
        else:
            visible_tokens = attention_mask.bool() & ~mask

        visible_sequences = []
        visible_lengths = []
        for row_hidden, row_visible in zip(hidden, visible_tokens):
            current = row_hidden[row_visible]
            visible_sequences.append(current)
            visible_lengths.append(current.shape[0])

        max_visible_len = max(visible_lengths)
        padded_visible = hidden.new_zeros(
            hidden.shape[0],
            max_visible_len,
            hidden.shape[-1],
        )
        visible_padding = torch.ones(
            hidden.shape[0],
            max_visible_len,
            dtype=torch.bool,
            device=hidden.device,
        )

        for row_index, current in enumerate(visible_sequences):
            current_len = current.shape[0]
            padded_visible[row_index, :current_len] = current
            visible_padding[row_index, :current_len] = False

        encoded_visible = self.encoder(
            padded_visible,
            src_key_padding_mask=visible_padding,
        )
        encoded = hidden.new_zeros(hidden.shape)
        for row_index, row_visible in enumerate(visible_tokens):
            current_len = visible_lengths[row_index]
            encoded[row_index, row_visible] = encoded_visible[row_index, :current_len]

        return encoded

    def _masked_mse(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return reconstruction.new_zeros(())
        return F.mse_loss(reconstruction[mask], target[mask])
