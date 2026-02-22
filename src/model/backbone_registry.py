"""
Shared backbone registry for all training, finetuning, and evaluation scripts.

Maps string identifiers to model classes.  Each class exposes the finetuning-
compatible API used throughout the PULSE codebase:
  - propose_masking(batch_size, num_patches, mask_ratio, device)
  - forward_encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep)
  - num_patches, modality_name, encoder

Models that support masked-reconstruction pretraining additionally expose:
  - forward_decoder(private, shared, ids_restore)
  - reconstruction_loss(target, pred, mask)
"""

from src.model.CheapSensorMAE import CheapSensorMAE
from src.model.ResNet1D import ResNet1DMAE
from src.model.TCN import TCNMAE
from src.model.ContrastiveEncoder import ContrastiveModel
from src.model.VICRegEncoder import VICRegModel
from src.model.AutoregressiveTransformer import AutoregressiveModel

# Full registry: every backbone that can be used for finetuning / evaluation
BACKBONE_REGISTRY = {
    'transformer': CheapSensorMAE,
    'resnet1d': ResNet1DMAE,
    'tcn': TCNMAE,
    'contrastive': ContrastiveModel,
    'vicreg': VICRegModel,
    'autoregressive': AutoregressiveModel,
}

# Subset of backbones that support masked-reconstruction pretraining
MAE_BACKBONE_REGISTRY = {
    'transformer': CheapSensorMAE,
    'resnet1d': ResNet1DMAE,
    'tcn': TCNMAE,
}


def get_backbone_class(name: str):
    """Look up a backbone class by name.  Raises ValueError if not found."""
    cls = BACKBONE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from: {sorted(BACKBONE_REGISTRY.keys())}"
        )
    return cls
