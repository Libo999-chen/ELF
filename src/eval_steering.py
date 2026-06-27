#!/usr/bin/env python
"""SM-ELF C2 steering eval (single device, requires an M2 checkpoint, manifold_dim>0).

Pipeline:
  1. Encode val stories -> pooled latent -> model.reencode -> code mu (=c).
  2. Label each story's sentiment (lexicon by default, or HF classifier) and
     fit a sentiment AXIS u in code space (difference-of-means).
  3. Sweep alpha: c = c0 + alpha*u, lift phi = U c, generate, classify.
  4. Report positive-fraction vs alpha. Monotonic with a large endpoint delta
     => controllability (C2). The cycle loss is what makes it faithful.

Trick: generation feeds a *steered* phi=U c directly. We run it through a
manifold_dim=0 VIEW of the same trained params (phi used as the conditioning
vector as-is, ManifoldCode bypassed), so no extra sampling plumbing is needed.

Usage:
  python3 src/eval_steering.py \
      --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
      --checkpoint_path /mnt/faster3/lc2762/elf_tinystories_sm_m2_output/checkpoint_XXXX \
      --label-stories 200 --samples-per-alpha 24 --alphas -3,-2,-1,0,1,2,3
"""

import argparse
import copy
import json
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
from utils.encoder_utils import encode_text
from utils.semantic_utils import compute_phi
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos
from configs.config import load_config_from_yaml


POS_WORDS = set("happy happily smiled smile smiles laugh laughed laughing fun joy joyful "
                "love loved loves kind friend friends friendly played play plays glad excited "
                "wonderful nice good great greatly beautiful proud hug hugged safe won win "
                "yummy delicious magic magical brave cheer cheered cheerful warm gentle".split())
NEG_WORDS = set("sad sadly cried cry crying scared afraid angry mad hurt hurts pain bad lonely "
                "lost fear feared worried worry upset broken sorry fight fought hate hated dark "
                "cold sick fell falling danger dangerous cruel mean scary terrible awful "
                "frightened grumpy nasty trouble".split())


def lexicon_sentiment(text):
    toks = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    p = sum(t in POS_WORDS for t in toks)
    n = sum(t in NEG_WORDS for t in toks)
    return p - n  # >0 positive, <0 negative


def _pad_batch(list_of_ids, max_length, pad_id):
    B = len(list_of_ids)
    ids = np.full((B, max_length), pad_id, dtype=np.int32)
    valid = np.zeros((B, max_length), dtype=np.float32)
    for i, seq in enumerate(list_of_ids):
        seq = list(seq)[:max_length]
        ids[i, : len(seq)] = seq
        valid[i, : len(seq)] = 1.0
    return jnp.asarray(ids), jnp.asarray(valid)


def _encode(ids, valid, encoder_apply_fn, encoder_params, cfg):
    enc_mask = jnp.broadcast_to(valid[:, None, :], (valid.shape[0], valid.shape[1], valid.shape[1]))
    return encode_text(
        input_ids=ids, attention_mask=enc_mask,
        encoder_apply_fn=encoder_apply_fn, encoder_params=encoder_params,
        latent_mean=cfg.latent_mean, latent_std=cfg.latent_std,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--label-stories", type=int, default=200)
    p.add_argument("--samples-per-alpha", type=int, default=24)
    p.add_argument("--alphas", type=str, default="-3,-2,-1,0,1,2,3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config_from_yaml(args.config)
    if not (cfg.semantic_factorization and cfg.manifold_dim > 0):
        sys.exit("eval_steering requires an M2 model (semantic_factorization=true, manifold_dim>0).")
    alphas = [float(a) for a in args.alphas.split(",")]
    sc = cfg.sampling_configs[0]
    steps = sc.num_sampling_steps[0] if isinstance(sc.num_sampling_steps, list) else sc.num_sampling_steps
    sccfg = sc.self_cond_cfg_scales[0] if isinstance(sc.self_cond_cfg_scales, list) else sc.self_cond_cfg_scales

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    pad_id = get_pad_token_id(tok, cfg.pad_token)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 1
    L, d = cfg.max_length, None

    enc_cfg, enc_model, _ = get_encoder(cfg.encoder_model_name, jnp.float32)
    enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
    d = enc_cfg.d_model

    # Two model views over the SAME params: m2 (manifold on) for reencode,
    # m0 (manifold off) for generation with a directly-supplied phi=U c.
    def build(manifold_dim):
        return ELF_models[cfg.model](
            text_encoder_dim=d, max_length=L,
            attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
            num_time_tokens=cfg.num_time_tokens,
            num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
            vocab_size=tok.vocab_size, num_model_mode_tokens=cfg.num_model_mode_tokens,
            num_phi_tokens=cfg.num_phi_tokens, manifold_dim=manifold_dim,
            bottleneck_dim=cfg.bottleneck_dim,
        )
    m2, m0 = build(cfg.manifold_dim), build(0)

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
    # m0 view: drop the (unused) manifold submodule params to avoid unexpected keys.
    m0_params = {k: v for k, v in params.items() if k != "manifold"}
    U = np.asarray(params["manifold"]["lift"]["kernel"])  # (k, d)
    log_for_0(f"Loaded M2 checkpoint; lift U: {U.shape}")

    # --- codes + sentiment labels on val stories ---
    val = load_dataset_split(cfg.eval_data_path)
    N = min(args.label_stories, len(val))
    raw = [val[i]["input_ids"] for i in range(N)]
    texts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw]
    ids, valid = _pad_batch(raw, L, pad_id)
    # batched encode (chunks) to bound memory
    mus = []
    B = 64
    for s in range(0, N, B):
        x0 = _encode(ids[s:s + B], valid[s:s + B], enc_model.apply, enc_params, cfg)
        pooled = compute_phi(x0, valid[s:s + B])[:, 0, :]
        _, mu, _ = apply_manifold_code(params["manifold"], pooled, cfg.manifold_dim, d)
        mus.append(np.asarray(mu))
    mu = np.concatenate(mus, axis=0)  # (N, k)
    labels = np.array([lexicon_sentiment(t) for t in texts])
    pos, neg = mu[labels > 0], mu[labels < 0]
    if len(pos) < 5 or len(neg) < 5:
        log_for_0(f"WARNING: few labeled examples (pos={len(pos)}, neg={len(neg)}); axis may be noisy.")
    u = pos.mean(0) - neg.mean(0)
    u = u / (np.linalg.norm(u) + 1e-8)
    c0 = mu.mean(0)
    log_for_0(f"codes mu: {mu.shape} | pos={len(pos)} neg={len(neg)} | axis ||u||=1")

    # --- steering sweep ---
    M = args.samples_per_alpha
    rows, results = [], []
    for a in alphas:
        c = (c0 + a * u).astype(np.float32)
        phi_lift = jnp.asarray(np.repeat((c @ U)[None, :], M, axis=0))  # (M, d)
        rng, nrng, trng = jax.random.split(rng, 3)
        z = jax.random.normal(nrng, (M, L, d)) * cfg.denoiser_noise_scale
        t_steps = get_sampling_steps(trng, n_steps=steps, time_schedule=sc.time_schedule,
                                     P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)
        latent = _generate_samples_single_batch(
            model_params=m0_params, model_apply_fn=m0.apply, rng=nrng,
            z=z, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
            config=cfg, sampling_config=sc, cfg_scale=1.0, self_cond_cfg_scale=sccfg, phi=phi_lift,
        )
        pred = np.asarray(mask_after_eos(_dlm_decode_batch(
            z=latent, model_params=m0_params, model_apply_fn=m0.apply,
            t_final_val=float(t_steps[-1]), config=cfg, self_cond_cfg_scale=sccfg, phi=phi_lift,
        ), eos_id, pad_id))
        gtexts = [tok.decode(pred[m], skip_special_tokens=True) for m in range(M)]
        scores = [lexicon_sentiment(g) for g in gtexts]
        pos_frac = float(np.mean([s > 0 for s in scores]))
        results.append((a, pos_frac))
        for g in gtexts:
            rows.append({"alpha": a, "generated": g})

    print("\n" + "=" * 56)
    print("C2 STEERING (sentiment axis)")
    print("=" * 56)
    print("  alpha   positive_fraction")
    for a, pf in results:
        print(f"  {a:+5.1f}   {pf:.3f}   {'#' * int(pf * 30)}")
    fracs = [pf for _, pf in results]
    delta = fracs[-1] - fracs[0]
    # monotonic non-decreasing (allow small dips)
    mono = all(fracs[i + 1] >= fracs[i] - 0.05 for i in range(len(fracs) - 1))
    verdict = (delta >= 0.3) and mono
    print(f"\n  endpoint delta = {delta:+.3f} | monotonic(+/-0.05) = {mono}")
    print(f"  VERDICT: {'PASS — interpretable, monotonic control' if verdict else 'FAIL — flat / non-monotonic'}")
    print("=" * 56 + "\n")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint_path.rstrip('/')) or '.', "steering_samples.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} steered generations to {out}.")


if __name__ == "__main__":
    main()
