#!/bin/bash
# Sample the same seed at several inference-step counts for a like-for-like comparison.
# Run in a GPU container.
CKPT=${CKPT:-stage0_uncond_out/checkpoint-60000}
N=${N:-4}
for S in 50 100 200; do
  python scripts/sample_stage0_uncond.py \
    --checkpoint "$CKPT" \
    --num_samples "$N" --batch_size "$N" \
    --num_inference_steps "$S" \
    --seed 0 \
    --out_dir "${CKPT}/samples_steps${S}"
done
