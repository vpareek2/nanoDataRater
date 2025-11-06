## 0) Goal & Non‑Goals

**Goal:** Implement a meta‑learned **DataRater** (R(x;φ)) that assigns **per‑example scores/weights** used to train a small language model (M(θ)). Optimize (φ) so that training (M) with those weights improves **held‑out validation** performance.

**Non‑Goals:** Matching original paper scale or numbers; training large LMs; building production infra. This is an algorithmic repro.

---

## 1) High‑Level Design

* **Inner loop (train model):** Train a small causal LM (M(θ)) on batches from `Train_A` with **weighted loss** using scores from (R(x;φ)). Use **truncated** steps (do not converge).
* **Outer loop (meta‑update rater):** Evaluate (M) on `Val`; compute **meta‑gradients** (∇*φ \mathcal{L}*{val}) **through** the inner updates; update (φ).
* **Population (optional but recommended):** Run **K** parallel/serial inner runs per meta‑round for stability (different inits or data shards), average meta‑grads.
* **Deployment (after training):** Freeze (φ^*). Use (R(\cdot;φ^*)) to **filter** (drop bottom‑K per batch) or **re‑weight** during final training.

---

## 2) Data & Splits

* **Corpus:** any tokenized pre‑training text (e.g., a small public subset).
* **Splits (disjoint):**

  * `Train_A` — used in inner loop updates of (M).
  * `Val` — used **only** for the outer meta objective (no leakage).
  * (Optional) `Train_B` — if you want to augment rater pretext losses; not required.
* **Granularity:** score per **sequence** (e.g., per 1k‑token chunk). If documents are long, take a fixed window for the rater.

---

## 3) Models

### 3.1 DataRater (R(x;φ)) — small encoder → scalar score

* **Inputs:** token ids for a short sequence (e.g., 256–512 tokens).
* **Architecture (reference):**

  * Token embedding → small **Transformer encoder** (e.g., 6–8 layers, d_model 384, heads 6, ff 1536) → pooled [CLS] (or mean) → MLP → scalar.
  * **Activation:** `sigmoid(z/τ)` for temperature τ (default τ=1.0) to map into (0,1).
* **Regularizers:**

  * **Score clamp:** (w_i ← \mathrm{clamp}(w_i, w_{min}, w_{max})) (e.g., 0.05–1.0).
  * **Entropy penalty** on scores to avoid collapse (optional).

### 3.2 Inner LM (M(θ)) — tiny causal LM

* **Inputs:** token ids for language modelling (same tokenizer).
* **Architecture (reference):**

  * Small **Transformer decoder** (e.g., 12 layers, d_model 512, heads 8, ff 2048), vocab ~50k, context len 1024.
  * Use standard cross‑entropy with teacher forcing.

---

## 4) Training Objectives

### 4.1 Weighted inner loss (per batch)

For batch (B={(x_i)}):

* Scores: (w_i = R(x_i;φ))
* LM loss: (\ell_i = \mathrm{CE}(M(x_i))) (token‑level CE averaged per example)
* **Weighted loss:** (L_{train} = \frac{1}{|B|} \sum_i w_i ,\ell_i)

### 4.2 Meta objective (outer)

* After **T_inner** inner steps, compute validation loss:

  * (L_{val} = \mathrm{CE}(M(θ_T), \text{Val batch}))
* **Meta‑update:** (φ \leftarrow φ - η_{outer} , ∇*φ L*{val}),
  differentiating **through the inner updates** that depended on (w=R(x;φ)).

---

## 5) Algorithm Controls (defaults)

| Hyperparameter             |           Default | Notes                             |
| -------------------------- | ----------------: | --------------------------------- |
| `K` (population size)      |                 2 | # of inner runs per meta‑round    |
| `T_inner` (steps)          |             1k–2k | truncated steps (do not converge) |
| Inner batch size           | 64 seqs × 1k toks | adjust to memory                  |
| Val batches per meta‑round |              8–16 | average for stability             |
| Keep rate (for deployment) |           0.5–0.7 | for online filtering              |
| Score clamp                |       [0.05, 1.0] | avoid degenerate zeros            |
| Rater temp τ               |               1.0 | lower → sharper weights           |
| Inner opt                  |       AdamW, 1e‑3 | weight decay 0.01                 |
| Rater opt                  |       AdamW, 1e‑4 | weight decay 0.01                 |
| Grad clip (inner/outer)    |               1.0 | per‑step clip                     |
| Mixed precision            |              bf16 | `torch.cuda.amp.autocast`         |
| Checkpointing              |                on | gradient checkpoint LM blocks     |

---

## 6) Implementation Plan (PyTorch)

### 6.1 Project layout

```
datarater/
  configs/
    base.yaml
  data/
    datamodule.py
  models/
    rater.py
    tiny_lm.py
  train/
    inner_step.py         # differentiable inner updates
    meta_trainer.py       # outer loop controller
    filtering.py          # deploy-time bottom-K
  utils/
    schedule.py
    logging.py
    seed.py
  scripts/
    run_meta.py           # entrypoint
    score_offline.py      # dump scores for a split
```

### 6.2 Data module (`data/datamodule.py`)

* Tokenizer; dataset sharding; bucketing by length.
* Dataloaders:

  * `train_loader_A`: returns `(input_ids, attn_mask)` for LM; plus a **short rater view** (e.g., first 256 tokens) to score.
  * `val_loader`: held‑out sequences for validation.
* Ensure disjointness between `Train_A` and `Val`.

### 6.3 Models

* `models/tiny_lm.py`: small GPT‑like module with gradient checkpointing toggles.
* `models/rater.py`: encoder that returns `w ∈ (0,1)` per sequence; includes temp τ, clamp bounds.

### 6.4 Differentiable inner loop (`train/inner_step.py`)

Two supported approaches:

**A) Pure autograd (no extra deps):**

* Keep a **list of parameters** `θ` and do functional updates:

  ```python
  grads = torch.autograd.grad(L_train, θ, create_graph=True)
  θ = [p - η_inner * g for p, g in zip(θ, grads)]
  ```
* Wrap LM forward in a **functional** style that accepts an explicit `θ`.

**B) `higher` (cleaner):**

* Use `higher.innerloop_ctx(model, opt, copy_initial_weights=True)` to get a **differentiable optimizer** that updates shadow params; accumulate meta‑gradients wrt rater params.

Either path must:

* Retain graph across **T_inner** steps (truncated unroll).
* Use AMP to keep memory in check.
* Optionally checkpoint LM blocks to reduce memory.

### 6.5 Outer loop (`train/meta_trainer.py`)

* For each **meta‑round**:

  1. `meta_grads = 0`
  2. For `k in 1..K`:

     * Initialize or lightly‑warmstart a fresh inner model (M^{(k)}).
     * Run **T_inner** differentiable steps over batches from `Train_A`:

       * Compute `w = R(x; φ)` (no detach), weighted CE, differentiable update of `θ`.
     * Compute `L_val` over several `val_loader` batches using current `θ_T`.
     * Accumulate `∇_φ L_val` into `meta_grads`.
  3. Update rater params once with `meta_grads / K`.
  4. Log: `L_train`, `L_val`, score stats (mean, entropy), grad norms.
  5. Every N rounds, evaluate **ablation** baselines (see §9).

* **EMA** of rater params is recommended for stability (store φ_ema for deployment).

### 6.6 Deployment / filtering (`train/filtering.py`)

* **Online:** during final LM training, for each micro‑batch:

  * Compute `w = R(x; φ*)`, create `keep_mask = topk_mask(w, keep_rate)`, drop bottom‑K samples, proceed with forward/backward.
* **Offline:** `scripts/score_offline.py` to score a split and write `scores.parquet` with `(example_id, score)`, then build a filtered index.

---

## 7) Pseudocode (end‑to‑end)

```python
# setup
rater = DataRater(...).to(device)
outer_opt = AdamW(rater.parameters(), lr=1e-4, weight_decay=0.01)

for meta_round in range(R):
    rater.train()
    outer_opt.zero_grad()
    meta_grads = [torch.zeros_like(p) for p in rater.parameters()]

    for k in range(K):
        lm = TinyLM(...).to(device)
        lm.train()
        θ = list(lm.parameters())

        # optional: differentiable optimizer state; else manual updates
        for t in range(T_inner):
            batch = next(train_loader_A)
            lm_inputs = batch["lm_inputs"]      # 1k tokens
            r_inputs  = batch["rater_inputs"]   # 256 tokens

            with torch.cuda.amp.autocast():
                w = rater(r_inputs)                         # (B,)
                logits = lm(lm_inputs)
                token_ce = tokenwise_ce(logits, lm_inputs["targets"])  # (B,)
                L_train = (w * token_ce).mean()

            grads = torch.autograd.grad(
                L_train, θ, create_graph=True, retain_graph=False
            )
            θ = [p - eta_inner * g for p, g in zip(θ, grads)]
            # (If using higher, replace the manual update with diffopt.step(L_train))

        # validation objective
        L_val_total = 0.0
        for _ in range(N_val_batches):
            vb = next(val_loader)
            with no_grad_reparam(θ):                       # run lm with θ_T
                logits = functional_lm_forward(lm, θ, vb)
                L_val_total += ce_mean(logits, vb["targets"])
        L_val = L_val_total / N_val_batches

        # accumulate meta-grad wrt rater params
        gφ = torch.autograd.grad(L_val, rater.parameters())
        meta_grads = [mg + g for mg, g in zip(meta_grads, gφ)]

    # outer update
    for p, g in zip(rater.parameters(), meta_grads):
        p.grad = g / K
    outer_opt.step()
    # log metrics, maybe EMA rater
```

*(Helper `no_grad_reparam` can temporarily bind `θ` into the model for forward‑only val; or use `higher`’s context.)*

---

## 8) Configuration (YAML)

```yaml
experiment:
  seed: 17
  device: "cuda"
  precision: "bf16"

data:
  tokenizer: "gpt2"
  seq_len_lm: 1024
  seq_len_rater: 256
  train_a_path: "data/train_a/*.jsonl"
  val_path: "data/val/*.jsonl"
  batch_size: 64
  num_workers: 4

rater:
  layers: 8
  d_model: 384
  n_heads: 6
  d_ff: 1536
  tau: 1.0
  w_min: 0.05
  w_max: 1.0
  weight_decay: 0.01
  lr: 1.0e-4

lm:
  layers: 12
  d_model: 512
  n_heads: 8
  d_ff: 2048
  vocab_size: 50257
  weight_decay: 0.01
  lr: 1.0e-3
  grad_clip: 1.0
  checkpointing: true

meta:
  rounds: 5
  population_K: 2
  t_inner_steps: 1500
  val_batches: 8

deploy:
  keep_rate: 0.6
  mode: "filter"   # or "reweight"
```

---

## 9) Metrics, Checks, & Ablations

### Core metrics

* **Meta:** validation loss/perplexity after inner unroll (`L_val`), rater score stats (mean, std, entropy), % dropped.
* **Sanity:** if you **detach** `w` (no meta‑grad), rater shouldn’t improve; if you **shuffle** `w` across batch, performance should degrade; **constant** `w=1` recovers the baseline.

### Ablations (should run)

* No rater (uniform weights).
* Rater w/ **no meta** (train φ with auxiliary loss only) vs full meta.
* **Different keep rates** at deploy: 0.5 / 0.7 / 0.85.
* **Shorter/longer** `T_inner` (e.g., 500 vs 2k steps).
* **Population size** `K` = 1 vs 2–3.

---

## 10) Compute & Stability Notes

* Truncated inner training is **required**; do not try to converge (M).
* Memory control: use AMP (bf16), gradient checkpointing in LM, small rater context (≤256 tokens).
* If meta‑grads are noisy:

  * Increase `val_batches`.
  * Use EMA of rater params for deployment.
  * Add small L2 on scores (toward mean 0.5) or entropy reg.
* If scores collapse to zeros/ones:

  * Clamp (`w_min`), raise τ, add entropy penalty, or normalize scores per batch (divide by mean).

---

## 11) Minimal Success Criteria

* Over several meta rounds, `L_val` after inner unroll **decreases** versus the uniform‑weight baseline.
* In deploy mode, using the frozen rater to filter (keep 50–70%) yields **equal or better** validation loss for a given training budget relative to training on all data.

---

## 12) What to build last (optional)

* **Offline scoring service** (scores.parquet) + filtered sampler.
* **Bottom‑K per‑micro‑batch** hook for online filtering.
* **Distributed population** (K inner runs in parallel workers).
* **Implicit‑gradient / truncated‑Neumann** meta update (advanced).

---

## 13) Dependencies

* `torch>=2.3`
* `transformers` (optional, if you want a ready tokenizer)
* `datasets` (optional, for loading text)
* `higher` (optional, for differentiable optimizers)
* `wandb` or `tensorboard` for logging

---

## 14) Runbook (toy → LM)

1. **Toy smoke test (MNIST‑style)**

   * Replace LM with 2‑layer MLP; verify meta‑grad is non‑zero; rater scores change; `L_val` goes down.
2. **Tiny LM test**

   * Switch to small causal LM; `T_inner=500`, `K=1`, `R=3`; confirm stability.
3. **Proper LM repro**

   * `T_inner=1500`, `K=2`, `R=5`; deploy rater with keep‑rate 0.6 and show compute‑normalized win.
