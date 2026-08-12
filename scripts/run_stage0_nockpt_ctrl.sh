#!/bin/bash
# CONTROL RUN 2: gradient checkpointing OFF. Removing the text conditioning changed
# nothing (control 1 hit the same cliff at ~21.5k), and the signature -- loss steady
# at 0.2-0.36 while the gradient norm goes 0.2 -> 40 -- says the backward pass is
# producing numbers the forward pass does not agree with. Gradient checkpointing
# recomputes activations during backward, so it is the one thing in this setup that
# can corrupt gradients while leaving the loss intact. Memory is not a constraint
# here: the run peaks at 40 GB of 178 GB.
# (previous header) identical to v5 except the text conditioning is removed -- every
# sample gets the same fixed null prompt, exactly as the old 64k-step run did.
# Five runs have now hit the same cliff a few hundred steps apart regardless of
# learning rate, beta2, warmup or gradient guarding. Per-sample text embeddings
# are the largest difference from the configuration that survived 64k, so this
# isolates them: clear step 22000 and the text path is the cause; die anyway and
# it is exonerated and the 3-plane data becomes the suspect.
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
OUTPUT_DIR=${OUTPUT_DIR:-stage0_nockpt_ctrl}

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
  --max_train_steps 26000 \
  --learning_rate 2e-5 \
  --new_layer_lr_mult 4 \
  --max_grad_norm 0.5 \
  --loss_spike_factor 8 \
  --grad_spike_norm 10 \
  --grad_spike_abs 10 \
  --adam_beta2 0.95 \
  --resume_fresh_schedule \
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
  --lr_warmup_steps 1000 \
  --max_sequence_length 256 \
  --mixed_precision bf16 \
  --checkpointing_steps 2000 \
  --dataloader_num_workers 6 \
  --n_modalities 3 \
  --mask_loss_by_avail \
  --use_avail \
  --tracker_project_name ulfmri-stage0 \
  --run_name stage0_nockpt_ctrl \
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
