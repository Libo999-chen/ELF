# Mitigation design: decorrelating attributes at fixed code dimension

## Goal
Reduce off-target leakage at a **fixed** code dimension `k` while preserving
on-target controllability (sentiment `0->1`). If a cheap regularizer moves `k=64`
leakage from ~0.875 down toward the full-rank floor (~0.33) at little control cost,
the paper's arc closes: **observe -> measure mechanism -> explain -> fix**.

## Why "orthogonalize the lift U" is the wrong fix
The leakage lives in the **readout / data-correlation geometry of the code**, not
in the column geometry of the lift `U in R^{d x k}`. Orthonormalizing U's columns
re-bases `phi=Uc` but does not change how the *attributes* are arranged inside the
code `c`: two correlated attributes can still share a readout direction in `c`
after any orthonormal change of basis of U. Our probe measures interference in
**code space** (the `c`-side), so the fix must act there.

## What we actually measured (and should attack)
The Manifold Probe shows: as `k` shrinks, the **readout directions** of correlated
attributes (sentiment vs. gender) become more non-orthogonal (interference rises
0.003 -> 0.083), and that interference predicts leakage at `r=0.995`. So the direct
lever is: **penalize that interference during training.**

## Proposed regularizer (primary): batchwise readout decorrelation
On each batch, with codes `c_i in R^k` and attribute labels (sentiment `s_i`,
gender `g_i`), form the per-batch difference-of-means (or ridge-probe) axes in code
space:
```
u_sent = mean(c | s=+) - mean(c | s=-)
u_gen  = mean(c | g=+) - mean(c | g=-)
L_dec  = ( <u_sent, u_gen> / (||u_sent|| ||u_gen||) )^2     # squared cosine
```
Add `lambda_dec * L_dec` to the objective. This **directly minimizes the exact
interference our probe shows drives leakage** — a tight narrative: measured
quantity becomes the training signal. Differentiable, ~free (a few dot products on
the code), no extra network.

Generalization to >2 attributes: penalize the mean squared off-diagonal of the
Gram matrix of the (unit-normalized) per-attribute axes.

## Alternative / stronger regularizers (if the cosine penalty underperforms)
1. **HSIC independence**: penalize `HSIC(P_sent c, g)` — the dependence between the
   code's sentiment-subspace projection and the gender label — with an RBF kernel.
   Catches nonlinear sharing the linear cosine misses. ~O(B^2) per batch, fine at
   B=80.
2. **Adversarial probe**: a small adversary predicts gender from the
   sentiment-steering component of `c`; train the code to defeat it
   (gradient reversal). Strongest but adds a network + tuning.
3. **Per-attribute subspace allocation**: reserve disjoint code blocks for declared
   attributes (`c = [c_sent | c_gen | c_free]`) and supervise each block to carry
   only its attribute. Architectural, less elegant, but a clean ablation.

## The honest part: this is a research bet, not a guarantee
The attributes are **correlated in the data** (positive stories skew female). A
decorrelation loss fights that data statistic, so it can:
- (a) reduce leakage but **cost on-target control**, or
- (b) **relocate** entanglement (e.g., push it into nonlinear / higher-order
  structure the linear penalty doesn't see).
Therefore the deliverable is **not** "leakage solved" but a **leakage–control
tradeoff (Pareto) curve** as a function of `lambda_dec`: even a partial reduction
with a quantified control cost completes the story and is publishable. We will also
re-run the **probe** on the mitigated model to show interference actually dropped
(mechanism-level confirmation that the fix works *for the stated reason*).

## Minimal experiment
- Train `k=64` with `lambda_dec in {0, 0.1, 1, 3, 10}` (5 short runs, same budget).
- For each: continuous off-target leakage (logit-shift / AUC), on-target control
  delta, and probe interference. Plot leakage & control vs `lambda_dec`.
- Success criterion (for the strong version): a `lambda_dec` exists where leakage
  drops materially (e.g. 0.875 -> <=0.6) with control delta still `>=0.9`.
- Either way: report the curve and the probe-confirmed interference drop.

## Cost
5 short `k=64` runs (decorrelation is ~free per step) + the existing eval/probe
pipeline. No new infrastructure. Slots in after the k-sweep endpoints free GPUs.
