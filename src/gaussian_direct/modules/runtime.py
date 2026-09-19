"""Parameter-free replay of historical Dropout RNG consumption."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def replay_dropout_rng(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    *,
    probability: float,
    training: bool,
) -> None:
    """Advance the RNG exactly where a removed historical Dropout ran.

    Canonical E00P checkpoints do not retain parameters whose outputs never
    reached a scientific objective. Training replay still consumes the same
    Dropout masks so active modules receive the historical RNG stream.
    """

    probability = float(probability)
    if not training or probability <= 0.0:
        return
    placeholder = reference.new_zeros(shape)
    functional.dropout(placeholder, p=probability, training=True, inplace=False)
