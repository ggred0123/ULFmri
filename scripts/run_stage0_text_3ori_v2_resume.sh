#!/bin/bash
# Stage 0 v2: resume of v1 from its last healthy checkpoint (step 20000).
#
# v1 was clean to step 20k (loss 0.19, recognisable brains in all three planes) and
# then diverged: losses of 10-100x from ~20.8k, and it never came back -- the floor
# settled at 1.12 and the samples stayed pure noise through 60k. The data was not at
# fault (latents are well behaved, |z|max ~4.8, identical stats across planes), so
# this run drops the learning rate, tightens clipping, and adds the spike guard.
# Stage 0: 3T prior over ALL THREE planes, conditioned on the plane via text.
#
# Differences from run_stage0_uncond.sh:
#   - data is cut axial + coronal + sagittal (register_and_preprocess_3t.py)
#   - the slice orientation is the text prompt, "" is the dropout / CFG-negative row
#   - trains from the stock SD3 weights, NOT from the old axial checkpoint-64000
#
# Full pipeline (steps 1-3 are one-time; see the block at the bottom to run them):
#   1. register_and_preprocess_3t.py  -> slices  (CPU, hours)
#   2. precompute_latents.py          -> latents (GPU, ~1h)
#   3. precompute_text_embeds.py      -> prompt bank (GPU, minutes)
#   4. this script                    -> training
set -e
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[ -f "$(dirname "$0")/wandb_env.sh" ] && source "$(dirname "$0")/wandb_env.sh"

ROOT=${ROOT:-/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin}
# Stock SD3-medium. Already downloaded on this cluster, so no gated HF access needed.
SD3_CACHE=/NHNHOME/WORKSPACE/26msit001_A/cyoh/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers
PRETRAINED=${PRETRAINED:-$SD3_CACHE/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671}
LATENT_ROOT=${LATENT_ROOT:-$ROOT/data_stage0_3ori_latents}
TEXT_EMBEDS=${TEXT_EMBEDS:-$ROOT/data_stage0_3ori_latents/orientation_embeds.pt}
VAE_PATH=${VAE_PATH:-$ROOT/sd3_local/vae}
OUTPUT_DIR=${OUTPUT_DIR:-stage0_text_3ori_v2}

accelerate launch --mixed_precision=bf16 --main_process_port=29556 \
  training/train_stage0_uncond.py \
  --pretrained_model_name_or_path "${PRETRAINED}" \
  --latent_root "${LATENT_ROOT}" \
  --text_embeds_path "${TEXT_EMBEDS}" \
  --vae_path "${VAE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 100000 \
  --learning_rate 2e-5 \
  --new_layer_lr_mult 4 \
  --max_grad_norm 0.5 \
  --loss_spike_factor 8 \
  --resume_from_checkpoint ${RESUME:-stage0_text_3ori_v1/checkpoint-20000} \
  --weighting_scheme sigma_sqrt \
  --timestep_sampling sigma_uniform \
  --text_dropout_prob 0.1 \
  --axial_only webb \
  --out_of_plane_weight 0.25 \
  --validation_steps 1000 \
  --num_validation_samples 2 \
  --num_inference_steps 50 \
  --inference_shift 1.0 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --max_sequence_length 256 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --checkpointing_steps 2000 \
  --dataloader_num_workers 6 \
  --n_modalities 3 \
  --mask_loss_by_avail \
  --use_avail \
  --tracker_project_name ulfmri-stage0 \
  --run_name stage0_text_3ori_v2 \
  "$@"

# ---------------------------------------------------------------- data prep
# PREP=$ROOT/envs/ulfmri-prep/bin/python      # nibabel + SimpleITK
# GPU=$ROOT/envs/ulfmri-gpu/bin/python        # torch + diffusers
#
# $PREP data/register_and_preprocess_3t.py \
#     --medical_root $ROOT/medical \
#     --out_root     $ROOT/data_stage0_3ori \
#     --datasets kcl ulfenc webb \
#     --orientations axial coronal sagittal \
#     --res 256
#
# $GPU data/precompute_latents.py \
#     --data_root $ROOT/data_stage0_3ori \
#     --out_root  $ROOT/data_stage0_3ori_latents \
#     --vae_path  $ROOT/sd3_local/vae
#
# $GPU data/precompute_text_embeds.py \
#     --pretrained_model_name_or_path "$PRETRAINED" \
#     --scheme orientation \
#     --out $ROOT/data_stage0_3ori_latents/orientation_embeds.pt
