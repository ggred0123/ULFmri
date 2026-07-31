#!/bin/bash
# Stage 1: conditional 64mT -> 3T translation.
# Initializes from a Stage 0 (48ch unconditional) checkpoint, adds a zero-init
# condition branch (48 -> 96ch input), trains with modality + CFG dropout.
# Run in a GPU container.
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[ -f "$(dirname "$0")/wandb_env.sh" ] && source "$(dirname "$0")/wandb_env.sh"

PRETRAINED=${PRETRAINED:-stabilityai/stable-diffusion-3-medium-diffusers}
STAGE0_CKPT=${STAGE0_CKPT:-/home/rintern14/ymk/sd3-mri/stage0_uncond_v5/checkpoint-64000}
LATENT_ROOT=${LATENT_ROOT:-/home/rintern14/ymk/data_stage1_latents}
NULL_EMBEDS=${NULL_EMBEDS:-/home/rintern14/ymk/data_stage1_latents/enhance_embeds.pt}
OUTPUT_DIR=${OUTPUT_DIR:-stage1_cond_v1}
VAL_INPUT_ROOT=${VAL_INPUT_ROOT:-/home/rintern14/ymk/data_stage1_val_latents}

accelerate launch --mixed_precision=bf16 --main_process_port=29557 \
  training/train_stage1_cond.py \
  --pretrained_model_name_or_path "${PRETRAINED}" \
  --stage0_checkpoint "${STAGE0_CKPT}" \
  --latent_root "${LATENT_ROOT}" \
  --null_embeds_path "${NULL_EMBEDS}" \
  --val_input_root "${VAL_INPUT_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --resolution 256 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 60000 \
  --learning_rate 5e-5 \
  --new_layer_lr_mult 10 \
  --weighting_scheme sigma_sqrt \
  --timestep_sampling sigma_uniform \
  --n_modalities 3 \
  --mask_loss_by_avail \
  --use_avail \
  --flair_dropout_prob 0.2 \
  --cond_dropout_prob 0.1 \
  --guidance_scale 2.0 \
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
  --dataloader_num_workers 6 \
  "$@"
