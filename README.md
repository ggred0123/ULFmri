# ULFmri — 64mT → 3T MRI translation with SD3

Turning ultra-low-field (64 mT) brain MRI into 3T-like images with a rectified-flow
diffusion model, built on Stable Diffusion 3's MMDiT.

Portable 64 mT scanners are cheap and deployable, but their images are low-SNR,
low-resolution and low-contrast compared to a 3T clinical scan. This repo trains a
model that maps a low-field scan to its 3T counterpart across three modality slots
`[T1w, T2w, FLAIR]`, while coping with the fact that **no dataset has all of them**.

## The problem: ragged modality availability

Three sources, none complete, and input/target availability differ independently:

| Dataset | ULF input | 3T target | ULF↔3T alignment |
|---|---|---|---|
| **ulfenc** (50 subj) | T1, T2, FLAIR | T1, T2, FLAIR | already on the same grid |
| **webb** (10 paired + 53 input-only) | T1, T2, FLAIR | T1, T2, FLAIR | same grid |
| **kcl** (21 subj) | T1, T2 only, multi-orientation, different grids | T1, T2 | needs ULF→3T registration |

Everything is carried in a fixed 3-slot order `[T1w, T2w, FLAIR]`. A missing slot is
**copy-filled** from a random available modality, and two separate binary masks record
the truth:

- `input_avail` — which condition modalities are real. Fed to the model as a
  conditioning embedding so it can tell a real FLAIR from a copy-filled one.
- `target_avail` — which target modalities exist. Used **only** as the loss mask, so
  kcl (no FLAIR) contributes no FLAIR gradient instead of being supervised against a
  stand-in.

These are never interchanged. See [training_plan.md](training_plan.md) §3 and §9.

## Two-stage approach

**Stage 0 — unconditional 3T prior.** Fine-tune stock SD3-medium on 3T slices alone to
learn `p(y_3T)`, using all 3T data including subjects that have no low-field partner.
This is what teaches the model what a valid 3T brain looks like. Optionally conditioned
on slice orientation (axial / coronal / sagittal) through the text path.

**Stage 1 — conditional translation.** Load the Stage 0 weights and add the condition by
channel concatenation, InstructPix2Pix-style:

```
model_input = concat(noisy_target_48ch, condition_48ch)   # 96ch
prediction  = v_theta(model_input, t, input_avail)        # 48ch
loss        = masked_flow_loss(prediction, target, target_avail)
```

Only `pos_embed.proj` grows (48→96 input channels); the condition half is **zero-init**,
so at step 0 the conditional model reproduces the Stage 0 prior exactly and learns to use
the condition from there. Details in [stage1_design.md](stage1_design.md).

### Why 48 channels

Each modality is replicated to 3ch and VAE-encoded **on its own** → 3 × 16 = 48 latent
channels, concatenated `[T1_16 | T2_16 | FLAIR_16]`. Packing the three contrasts into one
RGB image and sharing a single 16ch latent was measured (`data/check_packing_vs_permodality.py`)
to cost a lot of fidelity — T1w 30.6 → 39.6 dB, T2w 33.2 → 37.4 dB in favour of
per-modality encoding. Three decorrelated contrasts do not fit in one latent.

## Repo layout

```
data/          preprocessing + VAE/text precompute
  register_and_preprocess_3t.py   3T intra-subject rigid MI registration -> Stage 0 slices
  preprocess_pairs.py             paired (ULF, 3T) slices; registers kcl ULF -> 3T grid
  preprocess_val_inputs.py        input-only ULF set (no target) for generalization checks
  precompute_latents.py           Stage 0 slices  -> 48ch latents
  precompute_latents_pairs.py     Stage 1 pairs   -> 48ch input + target latents
  precompute_text_embeds.py       prompt bank (orientation / source + "" null row)
  check_*.py                      VAE round-trip, shift factor, packing sanity checks

training/
  train_stage0_uncond.py          Stage 0 3T prior
  train_stage1_cond.py            Stage 1 conditional translation
  train_sd3_pix2pix.py            upstream UltraEdit trainer (kept as reference)

scripts/
  run_stage0_*.sh                 Stage 0 runs and ablations (see below)
  run_stage1_cond.sh              Stage 1 run
  sample_stage0_uncond.py         unconditional / orientation-prompted sampling
  sample_stage1.py                64mT -> 3T inference with two-axis CFG
  eval_volume_metrics.py          whole-volume PSNR/SSIM over every held-out slice
```

## Pipeline

```bash
PREP=$ROOT/envs/ulfmri-prep/bin/python   # nibabel + SimpleITK
GPU=$ROOT/envs/ulfmri-gpu/bin/python     # torch + diffusers

# --- Stage 0 data (one-time) ---
$PREP data/register_and_preprocess_3t.py --medical_root $ROOT/medical \
      --out_root $ROOT/data_stage0_3ori --datasets kcl ulfenc webb \
      --orientations axial coronal sagittal --res 256
$GPU  data/precompute_latents.py     --data_root $ROOT/data_stage0_3ori \
      --out_root $ROOT/data_stage0_3ori_latents --vae_path $ROOT/sd3_local/vae
$GPU  data/precompute_text_embeds.py --scheme orientation \
      --out $ROOT/data_stage0_3ori_latents/orientation_embeds.pt

# --- Stage 0 training ---
bash scripts/run_stage0_final.sh

# --- Stage 1 data ---
$PREP data/preprocess_pairs.py
$GPU  data/precompute_latents_pairs.py

# --- Stage 1 training + inference + eval ---
bash scripts/run_stage1_cond.sh
$GPU scripts/sample_stage1.py \
     --checkpoint stage1_cond_v3_text/checkpoint-3500 \
     --input_root $ROOT/data_stage1_val --out_dir samples_v3 \
     --null_embeds_path $ROOT/data_stage1_latents/enhance_embeds.pt \
     --guidance_scale 2.0 --num_samples 4
$GPU scripts/eval_volume_metrics.py \
     --checkpoints stage1_cond_v3_text/checkpoint-* \
     --out stage1_cond_v3_text/volume_metrics.jsonl
```

Latents and text embeddings are precomputed so the training loop never loads the VAE or
the text encoders — that is ~11 GB of VRAM saved and a much faster step.

## Training recipe

Settings that mattered, all reflected in `scripts/run_stage0_final.sh`:

- **Velocity-space loss.** x0-MSE with σ⁻² weighting is algebraically the same as a plain
  velocity MSE, but computing it directly in velocity space avoids forming a tiny `σ·dv`
  in bf16 and then multiplying by σ⁻², which destroys precision at small σ.
- **`--weighting_scheme sigma_sqrt` + `--timestep_sampling sigma_uniform`.** The default
  logit-normal weighting gives the low-noise / fine-detail band almost no gradient. σ⁻²
  cancels the σ² and restores a uniform velocity loss, which visibly revives sharpness.
- **Layer-wise learning rates.** The expanded I/O layers (`pos_embed.proj`, `proj_out`)
  are the only parts that were never genuinely pretrained — they were replicated from
  16ch weights — so they get `--new_layer_lr_mult 4`. The conditioning path
  (`time_text_embed` + the avail embedders) gets `--cond_lr_mult 0.05`; see below.
- **Classifier-free guidance on two axes** in Stage 1: the whole condition is dropped with
  p≈0.1 and the text prompt with p=0.1, giving independent `--guidance_scale` and
  `--text_guidance_scale` at inference. Defaults 2.0 / 1.0 won the held-out sweep.
- **Spike guards** (`--loss_spike_factor`, `--grad_spike_norm`, `--grad_spike_abs`) —
  see the next section.

### Training stability: what went wrong and what fixed it

Stage 0 v1 was clean to step 20k (loss 0.19, recognisable brains in all three planes) and
then diverged — 10–100× loss bursts from ~20.8k, a floor at 1.12, and pure noise through
60k. Five runs hit the same cliff within a few hundred steps of each other regardless of
learning rate, β₂, warmup or clipping.

Gradient clipping does not save you here: Adam normalises the update anyway, so a run of
bad directions still corrupts the moments. The guards therefore **refuse** the step rather
than rescale it, and the EMAs they compare against only track accepted steps so a spike
storm cannot drag the threshold up behind it.

Layer attribution at each spike found the culprit: `time_text_embed.text_embedder.linear_2`
owned **86–92%** of the gradient norm while the image path barely registered. The
conditioning path (pooled projection → `time_text_embed` → every block's AdaLN modulation)
sees an almost constant input — `avail` is `[1,1,0]` or `[1,1,1]` and there are only a
handful of prompts — so its gradient points the same way every step instead of averaging
out over the batch, and accumulates without bound. Splitting it into its own parameter
group at 1/20 the backbone LR is the fix.

A series of controls isolated this, and the scripts are kept:

| script | what it tests |
|---|---|
| `run_stage0_text_3ori.sh` (+ `_v2/_v3/_v5_resume`) | LR, β₂, warmup, guard variations |
| `run_stage0_notext_ctrl.sh` | is per-sample text conditioning the cause? (no) |
| `run_stage0_nockpt_ctrl.sh` | is gradient checkpointing corrupting the backward? (no) |
| `run_stage0_layerprobe.sh` | per-layer gradient attribution |
| `run_stage0_condlr.sh` | separate low LR for the conditioning path |
| `run_stage0_final.sh` | the run that produced `stage0_final/` |

**Known limitation of `stage0_final`.** `--cond_lr_mult 0.05` delayed the blow-up but did
not remove it; from ~30k onward the gradient guard refused ~98% of steps (72,489 of 80,000
total). Weight-drift between checkpoints confirms the model is effectively frozen after
40k — the 22k→100k change is 11.35% on `pos_embed.proj`, of which 11.26% had already
happened by step 40k. The samples are fine because the model had converged by then, but
`checkpoint-40000` and `checkpoint-100000` are near-identical and the last 60k steps of
compute bought nothing. If Stage 0 is rerun, either stop at 40k or fix the conditioning
path further.

## Results

Whole-volume PSNR/SSIM on 3 held-out **subjects** (508 slices; subject-level split, since
adjacent slices would leak). Reference images are `decode48(target latent)`, so these
numbers share the VAE ceiling with the training log.

| run | notes | PSNR (vol) | PSNR (slice) | SSIM |
|---|---|---:|---:|---:|
| `stage1_cond_v2` | baseline, null text | 18.17 | 19.23 | 0.697 |
| `stage1_cond_v2b_notext_6k` | text path removed | 18.03 | 19.33 | 0.702 |
| `stage1_cond_v3_text` | per-source text + text CFG | 18.19 | 19.35 | 0.701 |
| `stage1_cond_v4_wkcl` | + kcl pairs | **18.39** | **19.59** | **0.709** |
| `stage1_cond_v5_nostage0` | **ablation: no Stage 0 init** | 17.49 | 18.46 | 0.673 |

The last row is the point of the two-stage design: dropping the Stage 0 prior and training
the conditional model from stock SD3 costs ~0.9 dB and 0.036 SSIM.

Averaging independent draws helps, since diffusion returns a sample rather than the
posterior mean — `--num_samples 4` measured +1.2 dB for 4× the compute.

**These Stage 1 numbers predate `stage0_final`.** They were produced from the older
axial-only `stage0_uncond_v5/checkpoint-64000` prior; Stage 1 has not yet been rerun on
the 3-orientation Stage 0.

## Setup

Two environments, because the dependencies do not overlap:

- **`ulfmri-prep`** (CPU) — preprocessing: `nibabel`, `SimpleITK`, `numpy`, `scikit-image`
- **`ulfmri-gpu`** (GPU) — precompute, training, sampling: `torch`, `diffusers`,
  `transformers`, `accelerate`, `sentencepiece`, `wandb`

`requirements.txt` is inherited from upstream UltraEdit and pins an older stack
(`torch==2.3.0+cu118`, `accelerate==0.31.0`); it does **not** list `diffusers`, `nibabel`
or `SimpleITK`, so it is not sufficient on its own. Fix it before relying on it.

Stage 0 as configured trains at 256², effective batch 16 (4 × 4 accumulation) on a single
GPU, peaking around 40 GB.

## Provenance

Forked from [UltraEdit](https://github.com/HaozheZhao/UltraEdit) (Zhao et al., 2024) for
its SD3 InstructPix2Pix training scaffolding, which is where the channel-concatenation
conditioning and the SD3 text-encoding helpers come from. `training/train_sd3_pix2pix.py`
and `scripts/run_sft_512_*.sh` are the unmodified upstream files, kept for reference; the
MRI pipeline in `data/`, `training/train_stage{0,1}_*.py` and `scripts/run_stage*.sh` is
this project's own.

```bib
@misc{zhao2024ultraeditinstructionbasedfinegrainedimage,
      title={UltraEdit: Instruction-based Fine-Grained Image Editing at Scale},
      author={Haozhe Zhao and Xiaojian Ma and Liang Chen and Shuzheng Si and Rujie Wu and Kaikai An and Peiyu Yu and Minjia Zhang and Qing Li and Baobao Chang},
      year={2024},
      eprint={2407.05282},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.05282},
}
```
