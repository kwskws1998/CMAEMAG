from dataclasses import dataclass

from src.configs.constants import DLModelNames
from src.configs.models.dl.MAG import MAG
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class CMAEMAGEye(MAG):
    """Hydra configuration for the C-MAE-MAG-Eye model."""

    base_model_name: DLModelNames = DLModelNames.CMAE_MAG_EYE_MODEL

    cmae_mask_ratio: float = 0.25
    cmae_hidden_dim: int = 256
    cmae_decoder_dim: int = 256
    cmae_num_encoder_layers: int = 2
    cmae_num_decoder_layers: int = 1
    cmae_num_attention_heads: int = 4
    cmae_dropout: float = 0.1
    cmae_loss_weight: float = 0.1
    cmae_residual_scale: float = 0.5
    cmae_detach_text_condition: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.cmae_hidden_dim % self.cmae_num_attention_heads != 0:
            raise ValueError(
                'cmae_hidden_dim must be divisible by cmae_num_attention_heads'
            )
        if self.cmae_decoder_dim % self.cmae_num_attention_heads != 0:
            raise ValueError(
                'cmae_decoder_dim must be divisible by cmae_num_attention_heads'
            )
        if not 0.0 <= self.cmae_mask_ratio < 1.0:
            raise ValueError('cmae_mask_ratio must satisfy 0 <= ratio < 1')
        if self.use_fixation_report:
            raise ValueError('CMAEMAGEye currently expects token-aligned IA features')
