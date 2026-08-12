#!/usr/bin/env python
"""
Sample from a Stage 0 3T MRI prior checkpoint.

Mirrors training/train_stage0_uncond.py: stock SD3 MMDiT, flow-matching Euler
sampling, optional avail embedding added to the pooled projection.

Text: pass --text_embeds_path (the orientation bank) + --prompt axial|coronal|
sagittal to ask for a plane. Without them every sample uses the null prompt,
which reproduces the older unconditional checkpoints.

Outputs, per sample: a .npy of the raw [3,H,W] float image (T1w/T2w/FLAIR in
[0,1]) plus a PNG grid of the three channels side by side.

Run in a GPU container:
  python scripts/sample_stage0_uncond.py \
    --checkpoint stage0_text_3ori_v1/checkpoint-60000 \
    --text_embeds_path data_stage0_3ori_latents/orientation_embeds.pt \
    --prompt coronal --num_samples 8
"""
import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel


class AvailEmbedder(torch.nn.Module):
    """Must match the trainer's definition for state_dict loading."""
    def __init__(self, pooled_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 256), torch.nn.SiLU(), torch.nn.Linear(256, pooled_dim)
        )

    def forward(self, avail):
        return self.net(avail)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a checkpoint-N dir (has transformer/)")
    p.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-3-medium-diffusers",
                   help="source of the VAE + scheduler config")
    p.add_argument("--null_embeds_path", default=None,
                   help="fixed null embeds; not needed if --text_embeds_path is given "
                        "(the bank carries its own '' row)")
    p.add_argument("--text_embeds_path", default=None,
                   help="prompt bank from data/precompute_text_embeds.py --scheme orientation")
    p.add_argument("--prompt", default=None,
                   help="row of the bank to sample, e.g. axial / coronal / sagittal. "
                        "Default: the null ('') row.")
    p.add_argument("--n_modalities", type=int, default=3,
                   help="latents are n_modalities x 16ch, ordered [T1|T2|FLAIR]; must match "
                        "the checkpoint. 1 = legacy packed-RGB 16ch latents.")
    p.add_argument("--out_dir", default=None, help="default: <checkpoint>/samples")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--avail", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="target_avail vector; [1,1,0] = FLAIR-missing (kcl-style), [1,1,1] = all modalities")
    p.add_argument("--no_avail", action="store_true", help="skip the avail embedder even if the checkpoint has one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out_dir = args.out_dir or os.path.join(args.checkpoint, "samples")
    os.makedirs(out_dir, exist_ok=True)

    mp = args.pretrained_model_name_or_path
    transformer = SD3Transformer2DModel.from_pretrained(
        args.checkpoint, subfolder="transformer", torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(mp, subfolder="vae", torch_dtype=dtype).to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(mp, subfolder="scheduler")

    if not (args.null_embeds_path or args.text_embeds_path):
        raise SystemExit("need --text_embeds_path (prompt bank) or --null_embeds_path")
    blob = torch.load(args.null_embeds_path or args.text_embeds_path, map_location="cpu")
    if args.null_embeds_path is None:
        row = args.prompt or "null"
        if row not in blob:
            raise SystemExit(f"--prompt {row!r} not in the bank; have {sorted(blob)}")
        print(f"prompt: {row!r} -> {blob[row].get('prompt', '')!r}")
        blob = blob[row]
    null_pe = blob["prompt_embeds"].to(device, dtype=dtype)
    null_pooled = blob["pooled_prompt_embeds"].to(device, dtype=dtype)

    avail_path = os.path.join(args.checkpoint, "avail_embedder.pt")
    avail_embedder = None
    if os.path.exists(avail_path) and not args.no_avail:
        avail_embedder = AvailEmbedder(null_pooled.shape[-1])
        avail_embedder.load_state_dict(torch.load(avail_path, map_location="cpu"))
        avail_embedder.to(device, dtype=dtype).eval()
        print(f"loaded avail_embedder, avail={args.avail}")

    latent_ch = transformer.config.in_channels
    lat_h = lat_w = args.resolution // (2 ** (len(vae.config.block_out_channels) - 1))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    made = 0
    while made < args.num_samples:
        bsz = min(args.batch_size, args.num_samples - made)
        latents = torch.randn(bsz, latent_ch, lat_h, lat_w,
                              generator=generator, device=device, dtype=dtype)

        pe = null_pe.repeat(bsz, 1, 1)
        pooled = null_pooled.repeat(bsz, 1)
        if avail_embedder is not None:
            avail = torch.tensor([args.avail], device=device, dtype=dtype).repeat(bsz, 1)
            pooled = pooled + avail_embedder(avail)

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        for t in scheduler.timesteps:
            model_pred = transformer(
                hidden_states=latents,
                timestep=t.expand(bsz).to(device),
                encoder_hidden_states=pe,
                pooled_projections=pooled,
                return_dict=False,
            )[0]
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]

        latents = latents.to(dtype) / vae.config.scaling_factor
        if args.n_modalities > 1:
            # each modality owns its own 16ch block and was encoded as a separate
            # grayscale image, so it has to be decoded on its own too (the trainer's
            # validation does the same). Decoding all 48 channels at once is not valid.
            ch = latents.shape[1] // args.n_modalities
            images = torch.stack(
                [vae.decode(latents[:, k * ch:(k + 1) * ch], return_dict=False)[0].mean(1)
                 for k in range(args.n_modalities)], dim=1)                 # [B,3,H,W]
        else:
            images = vae.decode(latents, return_dict=False)[0]
        images = ((images.float() + 1.0) / 2.0).clamp(0, 1).cpu().numpy()   # [B,3,H,W] in [0,1]

        for i in range(bsz):
            idx = made + i
            np.save(os.path.join(out_dir, f"sample_{idx:03d}.npy"), images[i].astype("float32"))
            save_png(images[i], os.path.join(out_dir, f"sample_{idx:03d}.png"))
        made += bsz
        print(f"{made}/{args.num_samples} done", flush=True)

    print(f"wrote {args.num_samples} samples to {out_dir}")


def save_png(img, path):
    """img: [3,H,W] in [0,1] -> PNG with T1w | T2w | FLAIR side by side."""
    from PIL import Image
    strip = np.concatenate([img[c] for c in range(img.shape[0])], axis=1)
    Image.fromarray((strip * 255).astype("uint8")).save(path)


if __name__ == "__main__":
    main()
