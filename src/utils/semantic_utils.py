"""Helpers for SM-ELF: semantic-manifold factorization x = phi(s) + r.

M1 uses the masked mean of the clean latent as the semantic code phi(s).
Later milestones replace `compute_phi` with a low-rank manifold encoder.
"""

import jax.numpy as jnp


def compute_phi(x0, valid_mask):
    """Sentence-level semantic code phi(s) = masked mean of x0 over valid positions.

    Args:
        x0: (B, S, C) clean latent embeddings.
        valid_mask: (B, S) or (B, S, 1), 1 at positions that count toward the mean
            (valid, non-conditioning tokens).

    Returns:
        (B, 1, C) phi, broadcastable over the sequence axis.
    """
    m = valid_mask.astype(x0.dtype)
    if m.ndim == 2:
        m = m[:, :, None]
    denom = jnp.maximum(m.sum(axis=1, keepdims=True), 1.0)
    return (x0 * m).sum(axis=1, keepdims=True) / denom
