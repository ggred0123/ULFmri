"""
Diagnose whether blurry samples come from the VAE or from the transformer.

Encodes real preprocessed slices with the SD3 VAE and decodes them straight back,
then saves original-vs-reconstruction montages plus PSNR/MAE. If the round-trip is
already blurry, the VAE is the sharpness ceiling and training the transformer
longer cannot fix it.

Output: <out_dir>/roundtrip_<i>.png   rows: [orig T1|T2|FLAIR] over [recon T1|T2|FLAIR]
Run in a GPU container.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/rintern14/ymk/data_stage0_3t_reg")
    ap.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae")
    ap.add_argument("--out_dir", default="/home/rintern14/ymk/data_stage0_3t_reg/_preview/vae_check")
    ap.add_argument("--n", type=int, default=6, help="how many slices to check")
    ap.add_argument("--stride", type=int, default=997, help="spread samples across the dataset")
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, torch.float32).eval()
    items = [json.loads(l) for l in open(data_root / "metadata.jsonl")]
    print(f"VAE latent_ch={vae.config.latent_channels} scaling={vae.config.scaling_factor}")

    psnrs, maes = [], []
    for i in range(args.n):
        it = items[(i * args.stride) % len(items)]
        arr = np.load(data_root / it["path"]).astype("float32")            # [3,H,W] in [0,1]
        x = torch.from_numpy(arr)[None].to(device) * 2.0 - 1.0             # [1,3,H,W] in [-1,1]

        lat = vae.encode(x).latent_dist.sample()
        recon = vae.decode(lat).sample                                     # [1,3,H,W] in [-1,1]

        o = ((x[0] + 1) / 2).clamp(0, 1).cpu().numpy()
        r = ((recon[0] + 1) / 2).clamp(0, 1).cpu().numpy()
        mae = float(np.abs(o - r).mean())
        mse = float(((o - r) ** 2).mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        psnrs.append(psnr); maes.append(mae)

        top = np.concatenate([o[c] for c in range(3)], axis=1)
        bot = np.concatenate([r[c] for c in range(3)], axis=1)
        grid = np.concatenate([top, bot], axis=0)
        fp = out_dir / f"roundtrip_{i}_{it['dataset']}_{it['subject']}.png"
        Image.fromarray((grid * 255).astype("uint8")).save(fp)
        print(f"{fp.name}  PSNR={psnr:.2f}dB  MAE={mae:.4f}")

    print(f"\nmean PSNR={np.mean(psnrs):.2f}dB  mean MAE={np.mean(maes):.4f}")
    print("guide: >35dB = VAE is fine (blur is the model); <30dB = VAE is the bottleneck")


if __name__ == "__main__":
    main()
