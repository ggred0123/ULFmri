# Stage 1 — Conditional 64mT → 3T Translation (design)

Turns the Stage 0 unconditional 3T prior into a conditional model that maps a
low-field (64mT / ULF) scan to its 3T counterpart, following training_plan.md §5–§10
adapted to our **per-modality 48-channel** latent layout.

---

## 1. Relationship to Stage 0

Stage 0 learned `p(y_3T)` with a 48ch MMDiT (`[T1|T2|FLAIR]`, each 16ch).
Stage 1 keeps that exact backbone and **adds the condition by channel
concatenation**, initialized so it starts out behaving like the Stage 0 prior and
gradually learns to use the condition.

```
model_input = concat( noisy_target_48ch , condition_48ch )   # 96ch
prediction  = v_theta(model_input, t, input_avail)           # 48ch (target only)
loss        = masked_flow_loss(prediction, target, target_avail)
```

---

## 2. Model change (one layer, zero-init condition half)

Only the input patch-embed grows; the output head stays 48ch.

| layer | Stage 0 | Stage 1 | init |
|---|---|---|---|
| `pos_embed.proj` | Conv2d(48, 1536, 2, 2) | Conv2d(**96**, 1536, 2, 2) | ch[0:48] ← Stage 0 weights (noisy-target branch); ch[48:96] ← **0** (condition branch) |
| `proj_out` | Linear(1536, 4·48) | unchanged | ← Stage 0 |
| everything else (24 MMDiT blocks, avail embedders, text path) | — | unchanged | ← Stage 0 |

Because the condition half is zero, at step 0 the model output == Stage 0 output
(the condition has no effect yet). This is training_plan.md §5 exactly, and reuses
the same `expand_transformer_channels` machinery (generalized to take a separate
`n_cond` that zero-inits).

**Weights are loaded from the Stage 0 checkpoint**, not stock SD3.

---

## 3. Data — the real work is here

Stage 1 needs **paired** (ULF input, 3T target) slices, spatially aligned so that
channel-concat in latent space means "same anatomy at same pixel". Alignment
status differs by source (from the earlier audit):

| source | ULF input | 3T target | ULF↔3T aligned? |
|---|---|---|---|
| **ulfenc** (50) | T1,T2,FLAIR on 3T grid | T1,T2,FLAIR | ✅ already same grid |
| **webb** (10) | T1,T2,FLAIR (raw, on HF-T2 grid) | T1,T2,FLAIR (registered to HF-T2) | ✅ same grid |
| **kcl** (21) | T1,T2 only, **multi-orientation, different grids** | T1,T2 | ❌ **needs ULF→3T registration** |

So the new preprocessing step is **register each subject's ULF to its 3T target
grid**, reusing the SimpleITK rigid + Mattes-MI path already written (fixed = 3T
T2w, moving = ULF modality, B-spline final resample). ulfenc/webb pass through;
kcl actually registers.

Per paired slice we store:
- `input_latent`  : 48ch VAE latent of ULF `[T1|T2|FLAIR]` (missing → copy-fill)
- `target_latent` : 48ch VAE latent of 3T `[T1|T2|FLAIR]` (Stage 0 latents reused)
- `input_avail`   : which ULF modalities are real (kcl = [1,1,0])
- `target_avail`  : which 3T modalities are real (kcl = [1,1,0])

### Modality availability
- **input_avail** — kcl ULF has no FLAIR → [1,1,0]; webb/ulfenc → [1,1,1]. Missing
  input slot copy-filled (random choice of an available modality) per §6.
- **target_avail** — same as Stage 0; used only as the loss mask (§9).
- Note input_avail and target_avail are independent (a subject can have a modality
  in one but not the other), so both are stored per slice.

### Condition-side modality dropout (§7)
On complete-input samples (webb, ulfenc) we randomly drop condition modalities so
the model is robust to missing inputs — essential because at inference kcl only
provides T1+T2. Policy (§7): 40% keep all / 40% drop one / 20% drop two, always
keep ≥1. Dropout is applied **only to the condition**; the target stays full, so
the model learns to *reconstruct* the dropped modality.

---

## 4. Availability as conditioning

`input_avail` is fed to the model (not used as a loss mask) so it knows which
condition slots are real vs copy-filled — via a small zero-init MLP added to the
pooled projection, exactly like the Stage 0 `AvailEmbedder`. Stage 1 therefore has
two embedders (input_avail + target_avail); both start at zero.

`target_avail` continues to mask the loss per modality (kcl: no FLAIR gradient).

---

## 5. Loss & training recipe (reused from Stage 0)

Same flow-matching objective, and the same fixes we landed in Stage 0:
- velocity-space loss (numerically stable at small sigma)
- `--weighting_scheme sigma_sqrt`
- `--timestep_sampling sigma_uniform` (visit the detail band)
- per-modality masking by `target_avail`
- layer-wise LR: condition branch + avail embedders get the higher LR; the
  pretrained backbone (loaded from Stage 0) gets the low LR.

Text prompt stays null (or later: orientation/source tokens — orthogonal to this).

---

## 6. Classifier-free guidance (the payoff)

This is what conditional buys us over Stage 0's blur. During training we randomly
drop the **whole condition** (zero the condition latent + set input_avail signal to
"none") with prob ~0.1, so the model learns both `v(x|c)` and `v(x|∅)`. At
inference:

```
v = v_uncond + guidance_scale * (v_cond - v_uncond)
```

`guidance_scale > 1` sharpens and pushes samples toward the conditioned target —
directly countering the softness. (Distinct from the per-modality dropout in §3,
which drops individual modalities; this drops the entire condition.)

---

## 7. Sampling / inference

```
input: a 64mT scan (T1,T2[,FLAIR]) -> register to a target grid -> VAE encode -> cond 48ch
start:  y_T ~ N(0, I)  (48ch)
loop:   v = CFG( transformer(concat(y_t, cond), t, input_avail) );  y_{t-1} = step(v, y_t)
decode: each 16ch block -> T1_3T, T2_3T, FLAIR_3T
```
Reuse the Stage 0 sampler with: 96ch input assembly, a `--guidance_scale`, and
`--inference_shift` for the detail band.

---

## 8. Concrete build order

1. **preprocess_pairs.py** — for each subject with ULF+3T: register ULF→3T grid
   (SimpleITK, kcl only), slice axially, save paired `[3,H,W]` input+target + avails.
2. **precompute_latents_pairs.py** — VAE-encode input & target to 48ch (extend the
   existing per-modality encoder; target latents can be reused from Stage 0).
3. **train_stage1_cond.py** — fork of the Stage 0 trainer:
   - load transformer from Stage 0 ckpt, expand 48→96 (zero-init condition half)
   - dataset returns (input_latent, target_latent, input_avail, target_avail)
   - condition dropout (§3) + full-condition dropout for CFG (§6)
   - concat in the loop, masked loss, input_avail embedder
4. **sample_stage1.py** — CFG sampler above.
5. run script `run_stage1_cond.sh`.

---

## 9. Open decisions (need your call)

1. **kcl ULF orientation.** kcl ULF is 4 separate acquisitions (Axial / Coronal /
   Sagittal / TomoBrain), each a different grid, and some subjects lack a modality
   in a given orientation. Options:
   - (a) pick one orientation per subject (e.g. TomoBrain, which had T1&T2 co-gridded) and register it to 3T;
   - (b) register **all** available ULF orientations to 3T and treat each as a separate training pair (more data, more variety of input quality);
   - (c) use the existing `derivatives/` coregistration outputs instead of re-registering.
2. **Guidance dropout rate** (default 0.1) and **modality-dropout policy** (default §7's 40/40/20).
3. **Slice pairing axis** — axial only (as Stage 0), or add coronal/sagittal later.
4. Whether to also condition on **source/orientation** via the (currently null) text path.

---

## 10. Risks

- **kcl ULF→3T registration quality.** Low-field ULF is low-SNR; rigid MI to a 3T
  target can misalign. Bad pairs teach the model wrong correspondences. Needs visual QC (like the Stage 0 montages).
- **Domain/resolution gap in the condition.** 64mT upsampled to the 3T grid is
  blocky; the model must learn a large super-resolution + contrast-translation jump. Expected, but sets a difficulty floor.
- **kcl has no FLAIR on either side**, so kcl contributes no FLAIR-translation
  signal; FLAIR mapping is learned from webb+ulfenc only (60 subjects).
