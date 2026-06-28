"""Helpers for SM-ELF: semantic-manifold factorization x = phi(s) + r.

M1 uses the masked mean of the clean latent as the semantic code phi(s).
Later milestones replace `compute_phi` with a low-rank manifold encoder.
"""

import jax.numpy as jnp

# --- dependency-free lexicons (shared by training-time decorrelation + evals) ---
_POS = set("happy happily smiled smile smiles laugh laughed laughing fun joy joyful "
           "love loved loves kind friend friends friendly played play plays glad excited "
           "wonderful nice good great greatly beautiful proud hug hugged safe won win "
           "yummy delicious magic magical brave cheer cheered cheerful warm gentle".split())
_NEG = set("sad sadly cried cry crying scared afraid angry mad hurt hurts pain bad lonely "
           "lost fear feared worried worry upset broken sorry fight fought hate hated dark "
           "cold sick fell falling danger dangerous cruel mean scary terrible awful "
           "frightened grumpy nasty trouble".split())
_FEM = set("she her hers girl woman women mom mommy mother sister aunt grandma queen "
           "princess lady daughter".split())
_MAL = set("he him his boy man men dad daddy father brother uncle grandpa king prince guy son".split())


def lexicon_labels(text):
    """(sentiment, gender) in {-1,0,+1} from word-count sign. Used to form the
    code-space difference-of-means axes for the decorrelation regularizer."""
    toks = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    p = sum(t in _POS for t in toks); n = sum(t in _NEG for t in toks)
    f = sum(t in _FEM for t in toks); m = sum(t in _MAL for t in toks)
    sent = 1 if p > n else (-1 if n > p else 0)
    gen = 1 if f > m else (-1 if m > f else 0)
    return sent, gen


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
