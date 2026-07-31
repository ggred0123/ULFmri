"""
Is the pseudo-RGB packing itself costing us fidelity?

The SD3 VAE was trained on natural RGB, where channels are highly correlated. Our
channels are [T1w, T2w, FLAIR] -- three decorrelated contrasts, which the VAE may
treat like extreme chroma and low-pass. This compares, per channel:

  (a) PACKED     : [T1,T2,FLAIR] as one RGB image -> VAE -> per-channel PSNR
  (b) PER-MODALITY: each modality replicated to 3ch (grayscale) -> VAE -> PSNR

If (b) is much better than (a) for T1w/FLAIR, the packing is the bottleneck and
per-modality latents (3x16=48ch) would buy real sharpness.

Run in a GPU container.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL

MODS = ["T1w", "T2w", "FLAIR"]


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/rintern14/ymk/data_stage0_3t_reg")
    ap.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae")
    ap.add_argument("--out_dir", default="/home/rintern14/ymk/data_stage0_3t_reg/_preview/packing_check")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stride", type=int, default=997)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, torch.float32).eval()

    items = [json.loads(l) for l in open(data_root / "metadata.jsonl")]
    packed_ps = {m: [] for m in MODS}
    permod_ps = {m: [] for m in MODS}

    for i in range(args.n):
        it = items[(i * args.stride) % len(items)]
        arr = np.load(data_root / it["path"]).astype("float32")        # [3,H,W] in [0,1]

        # (a) packed: all three modalities as one RGB image
        x = torch.from_numpy(arr)[None].to(device) * 2 - 1
        rec = vae.decode(vae.encode(x).latent_dist.sample()).sample
        rec = ((rec[0] + 1) / 2).clamp(0, 1).cpu().numpy()

        # (b) per-modality: each channel replicated to 3ch, encoded on its own
        per = []
        for c in range(3):
            g = torch.from_numpy(arr[c])[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
            r = vae.decode(vae.encode(g).latent_dist.sample()).sample
            per.append(((r[0] + 1) / 2).clamp(0, 1).mean(0).cpu().numpy())  # back to 1ch

        for c, m in enumerate(MODS):
            packed_ps[m].append(psnr(arr[c], rec[c]))
            permod_ps[m].append(psnr(arr[c], per[c]))

        grid = np.concatenate([
            np.concatenate([arr[c] for c in range(3)], axis=1),
            np.concatenate([rec[c] for c in range(3)], axis=1),
            np.concatenate([per[c] for c in range(3)], axis=1),
        ], axis=0)
        Image.fromarray((grid * 255).astype("uint8")).save(
            out_dir / f"pack_{i}_{it['dataset']}_{it['subject']}.png")

    print(f"\n{'modality':<8} {'PACKED(RGB)':>12} {'PER-MODALITY':>14} {'gain':>8}")
    for m in MODS:
        a, b = np.mean(packed_ps[m]), np.mean(permod_ps[m])
        print(f"{m:<8} {a:>11.2f}dB {b:>13.2f}dB {b-a:>+7.2f}dB")
    print(f"\nmontages -> {out_dir}  (rows: original / packed-recon / per-modality-recon)")
    print("A large positive gain on T1w+FLAIR means the RGB packing is the bottleneck.")


if __name__ == "__main__":
    main()
