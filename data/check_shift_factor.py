"""
Does the SD3 shift_factor actually matter for our data?

We currently normalize latents as   x * scaling_factor
SD3's own convention is            (x - shift_factor) * scaling_factor

Test: feed both versions to the UNTRAINED stock SD3 transformer and compare the
flow-matching (velocity) loss. The pretrained model was trained on latents in its
own convention, so whichever version it scores better on is the distribution it
actually expects. Paired comparison -- identical noise and timesteps for both.

Also prints latent statistics, since the point of the shift is to centre them.

Run in a GPU container. Uses a single modality (16ch) so the stock 16-channel
transformer can be used unmodified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--data_root", default="/home/rintern14/ymk/data_stage0_3t_reg")
    ap.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae")
    ap.add_argument("--null_embeds_path", default="/home/rintern14/ymk/data_stage0_3t_latents/null_embeds.pt")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--modality", type=int, default=0, help="0=T1w 1=T2w 2=FLAIR")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    mp = args.pretrained_model_name_or_path

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, torch.float32).eval()
    scale, shift = vae.config.scaling_factor, vae.config.shift_factor
    print(f"scaling_factor={scale}  shift_factor={shift}")

    transformer = SD3Transformer2DModel.from_pretrained(mp, subfolder="transformer").to(device, dtype).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(mp, subfolder="scheduler")
    blob = torch.load(args.null_embeds_path, map_location="cpu")
    pe0 = blob["prompt_embeds"].to(device, dtype)
    pooled0 = blob["pooled_prompt_embeds"].to(device, dtype)

    items = [json.loads(l) for l in open(Path(args.data_root) / "metadata.jsonl")]
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    tot = {"no_shift": 0.0, "with_shift": 0.0}
    stats = {"raw_mean": [], "raw_std": []}
    n = 0

    for _ in range(args.n_batches):
        idx = rng.choice(len(items), args.batch_size, replace=False)
        arr = np.stack([np.load(Path(args.data_root) / items[i]["path"])[args.modality] for i in idx])
        x = torch.from_numpy(arr)[:, None].repeat(1, 3, 1, 1).to(device, torch.float32) * 2 - 1
        raw = vae.encode(x).latent_dist.sample()                  # un-normalized latent
        stats["raw_mean"].append(float(raw.mean()))
        stats["raw_std"].append(float(raw.std()))

        variants = {"no_shift": raw * scale, "with_shift": (raw - shift) * scale}

        # identical noise + timesteps for both variants (paired comparison)
        noise = torch.randn_like(raw)
        u = torch.rand(raw.shape[0])
        ti = (u * sched.config.num_train_timesteps).long()
        timesteps = sched.timesteps[ti].to(device)
        sig = sched.sigmas.to(device)[ti].to(device)[:, None, None, None]

        for k, lat in variants.items():
            lat = lat.to(dtype)
            noisy = sig.to(dtype) * noise.to(dtype) + (1 - sig.to(dtype)) * lat
            v_pred = transformer(
                hidden_states=noisy, timestep=timesteps,
                encoder_hidden_states=pe0.repeat(lat.shape[0], 1, 1),
                pooled_projections=pooled0.repeat(lat.shape[0], 1),
                return_dict=False)[0]
            v_true = noise.to(dtype) - lat
            tot[k] += float(((v_pred.float() - v_true.float()) ** 2).mean())
        n += 1

    print(f"\nraw latent  mean={np.mean(stats['raw_mean']):+.4f}  std={np.mean(stats['raw_std']):.4f}")
    print(f"  -> no_shift   normalized mean={(np.mean(stats['raw_mean']))*scale:+.4f}")
    print(f"  -> with_shift normalized mean={(np.mean(stats['raw_mean'])-shift)*scale:+.4f}  (closer to 0 is the intent)")
    print(f"\nstock SD3 velocity loss over {n} batches (lower = distribution the model expects):")
    for k in ("no_shift", "with_shift"):
        print(f"  {k:<11} {tot[k]/n:.4f}")
    d = (tot['no_shift'] - tot['with_shift']) / n
    print(f"\ndifference (no_shift - with_shift) = {d:+.4f}")
    print("positive => the shift helps; near zero => it does not matter for this data")


if __name__ == "__main__":
    main()
