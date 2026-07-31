#!/bin/bash
# Stage 0: unconditional 3T MRI prior (stock SD3 16ch, null text, flow matching).
# Run in a GPU container (heavy). Needs access to the SD3 checkpoint.
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# wandb key (gitignored secret file). Or export WANDB_API_KEY yourself before running.
[ -f "$(dirname "$0")/wandb_env.sh" ] && source "$(dirname "$0")/wandb_env.sh"

PRETRAINED=${PRETRAINED:-stabilityai/stable-diffusion-3-medium-diffusers}
DATA_ROOT=${DATA_ROOT:-/home/rintern14/ymk/data_stage0_3t_reg}
OUTPUT_DIR=${OUTPUT_DIR:-stage0_uncond_out}
# Precomputed latents (recommended for 46GB): set to /home/rintern14/ymk/data_stage0_3t_latents
LATENT_ROOT=${LATENT_ROOT:-}
LATENT_ARG=""
[ -n "$LATENT_ROOT" ] && LATENT_ARG="--latent_root ${LATENT_ROOT}"
# Precomputed null text embeds (recommended): avoids loading text encoders (~11GB).
NULL_EMBEDS=${NULL_EMBEDS:-}
NULL_ARG=""
[ -n "$NULL_EMBEDS" ] && NULL_ARG="--null_embeds_path ${NULL_EMBEDS}"

accelerate launch --mixed_precision=bf16 --main_process_port=29556 \
  training/train_stage0_uncond.py \
  --pretrained_model_name_or_path "${PRETRAINED}" \
  --data_root "${DATA_ROOT}" \
  ${LATENT_ARG} \
  ${NULL_ARG} \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 60000 \
  --learning_rate 5e-5 \
  --new_layer_lr_mult 10 \
  --weighting_scheme sigma_sqrt \
  --timestep_sampling sigma_uniform \
  --validation_steps 1000 \
  --num_validation_samples 4 \
  --num_inference_steps 50 \
  --inference_shift 1.0 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --max_sequence_length 256 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --checkpointing_steps 2000 \
  --n_modalities 3 \
  --mask_loss_by_avail \
  --use_avail \
  "$@"

# Smoke test first (tiny, few steps) to catch shape/API errors:
#   accelerate launch training/train_stage0_uncond.py \
#     --data_root /home/rintern14/ymk/data_stage0_3t_reg \
#     --train_batch_size 2 --max_train_steps 5 --checkpointing_steps 5 --output_dir /tmp/s0_smoke
