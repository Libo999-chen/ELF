#!/usr/bin/env python
"""Measure attribute superposition in the SM-ELF code space (Manifold Probe).

Loads an SM-ELF checkpoint, encodes val stories -> codes c (R^k for M2, R^d
mean-pool for M1), labels each story's attributes (sentiment / gender / animal /
length), and runs the Manifold Probe (arXiv:2605.18537) per attribute to recover
its encoding subspace in code space. Reports the cross-attribute interference
matrix, the mean off-diagonal interference vs. the random baseline (excess), and a
capacity ratio. Run across the k-sweep checkpoints to test whether superposition
*directly measured in code space* grows as k shrinks and tracks off-target leakage.

Usage:
  python3 src/eval_superposition.py \
      --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
      --checkpoint_path /mnt/faster3/.../checkpoint_XXXX \
      --config_override manifold_dim=16 --label-stories 600
"""

import argparse
import copy
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.t5_encoder import get_encoder
from modules.model import ELF_models, apply_manifold_code
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState
from utils.data_utils import load_dataset_split, get_pad_token_id
from utils.semantic_utils import compute_phi
from configs.config import load_config_from_yaml, apply_config_overrides

# reuse the exact labelers + encode plumbing from the steering eval
from eval_steering import (
    lexicon_sentiment, attr_scores, gender_label, _pad_batch, _encode,
    FEMALE_WORDS, MALE_WORDS, ANIMAL_WORDS,
)
from utils.manifold_probe import superposition_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=5,
                   help="Average the superposition metrics over this many train/test splits.")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if not cfg.semantic_factorization:
        sys.exit("eval_superposition requires an SM-ELF model (semantic_factorization=true).")
    is_m2 = cfg.manifold_dim > 0

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    L = cfg.max_length

    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
    d = enc_cfg.d_model

    m2 = ELF_models[cfg.model](
        text_encoder_dim=d, max_length=L,
        attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
        num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
        num_phi_tokens=cfg.num_phi_tokens, manifold_dim=cfg.manifold_dim,
        bottleneck_dim=cfg.bottleneck_dim,
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    input_dim = 2 * d if cfg.self_cond_prob > 0 else d
    params_init = m2.init(
        init_rng, x=jnp.ones((1, L, input_dim)), t=jnp.ones((1,)), deterministic=True,
        self_cond_cfg_scale=jnp.ones((1,)) if cfg.num_self_cond_cfg_tokens > 0 else None,
        phi=jnp.ones((1, d)),
    )
    state = TrainState.create(
        apply_fn=m2.apply, params=params_init["params"], tx=optax.adamw(1e-4),
        dropout_rng=rng, ema_params1=copy.deepcopy(params_init["params"]),
    )
    state, _ = load_checkpoint(args.checkpoint_path, state)
    params = state.ema_params1 if cfg.eval_use_ema else state.params
    log_for_0(f"Loaded checkpoint (manifold_dim={cfg.manifold_dim}, "
              f"{'M2 learned code' if is_m2 else 'M1 mean-pool'}).")

    # --- codes + attribute labels on val stories ---
    val = load_dataset_split(cfg.eval_data_path)
    N = min(args.label_stories, len(val))
    raw = [val[i]["input_ids"] for i in range(N)]
    texts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw]
    ids, valid = _pad_batch(raw, L, pad_id)
    mus, B = [], 64
    for s in range(0, N, B):
        x0 = _encode(ids[s:s + B], valid[s:s + B], enc_model.apply, enc_params, cfg)
        pooled = compute_phi(x0, valid[s:s + B])[:, 0, :]
        if is_m2:
            _, mu_b, _ = apply_manifold_code(params["manifold"], pooled, cfg.manifold_dim, d)
            mus.append(np.asarray(mu_b))
        else:
            mus.append(np.asarray(pooled))     # M1: code = d-dim mean-pool phi
    C = np.concatenate(mus, axis=0)            # (N, k)  codes
    dim = C.shape[1]

    # attribute labels (scalar). sentiment & length are ~continuous; gender & animal binary.
    def _toks(t):
        return [w.strip(".,!?;:\"'").lower() for w in t.split()]
    sent = np.array([lexicon_sentiment(t) for t in texts], dtype=float)
    gend = np.array([gender_label(t) for t in texts], dtype=float)          # -1/0/+1
    animal = np.array([float(sum(w in ANIMAL_WORDS for w in _toks(t)) > 0) for t in texts])
    length = np.array([float(len(_toks(t))) for t in texts])
    labels = {"sentiment": sent, "gender": gend, "animal": animal, "length": length}

    # quick label-correlation context (sentiment<->gender is the steering pair)
    def _corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        return float((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    log_for_0(f"codes {C.shape} | label corr: sent-gender={_corr(sent, gend):+.3f} "
              f"sent-animal={_corr(sent, animal):+.3f} sent-length={_corr(sent, length):+.3f}")

    # average the metrics over several train/test splits (encode once, re-split)
    reps = [superposition_report(C, labels, dim=dim, degree=5, seed=args.seed + s)
            for s in range(max(1, args.seeds))]
    names = reps[0]["concepts"]
    si, gi = names.index("sentiment"), names.index("gender")

    def ms(key_fn):
        vals = np.array([key_fn(r) for r in reps], dtype=float)
        return float(vals.mean()), float(vals.std())

    mo = ms(lambda r: r["mean_offdiag_interference"])
    ex = ms(lambda r: r["excess_interference"])
    sg = ms(lambda r: r["interference"][si, gi])
    cap = ms(lambda r: r["capacity_ratio"])
    interf_mean = np.mean([r["interference"] for r in reps], axis=0)
    mdim_mean = {n: ms(lambda r, n=n: r["probes"][n]["manifold_dim"]) for n in names}

    print("\n" + "=" * 70)
    print(f"SUPERPOSITION PROBE (code space)  manifold_dim={dim}  "
          f"({'M2 learned' if is_m2 else 'M1 mean-pool'})  N={C.shape[0]}  seeds={args.seeds}")
    print("=" * 70)
    print("per-concept manifold dimension (mean over seeds):")
    for n in names:
        print(f"  {n:<10} mdim={mdim_mean[n][0]:.2f}+-{mdim_mean[n][1]:.2f}")
    print("\nmean interference matrix (subspace overlap in [0,1]):")
    print("           " + "".join(f"{n[:7]:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<8} " + "".join(f"{interf_mean[i, j]:>9.3f}" for j in range(len(names))))
    print(f"\n  MEAN OFF-DIAGONAL INTERFERENCE = {mo[0]:.4f} +- {mo[1]:.4f}")
    print(f"  random baseline (1/k)          = {1.0 / max(1, dim):.4f}")
    print(f"  EXCESS INTERFERENCE            = {ex[0]:+.4f} +- {ex[1]:.4f}   "
          f"(>0 => structured superposition above chance)")
    print(f"  sentiment<->gender interference= {sg[0]:.4f} +- {sg[1]:.4f}   (steering/leakage pair)")
    print(f"  CAPACITY RATIO (sum mdim / k)  = {cap[0]:.4f} +- {cap[1]:.4f}")
    print("=" * 70)
    print(f"SUPERPOSITION_SUMMARY k={dim} N={C.shape[0]} "
          f"mean_off={mo[0]:.4f} excess={ex[0]:.4f} sent_gender={sg[0]:.4f} "
          f"capacity={cap[0]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
