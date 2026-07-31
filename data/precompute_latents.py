"""
Precompute SD3 VAE latents for the Stage 0 dataset, so training never loads the
VAE (saves VRAM + speeds up the loop). Uses the LOCAL SD3 VAE, so this step needs
NO gated HuggingFace access.

PER-MODALITY encoding (default): each of [T1w, T2w, FLAIR] is replicated to 3
channels and encoded on its own -> 3 x 16 = 48 latent channels, concatenated as
[T1w_16 | T2w_16 | FLAIR_16].

Why not pack the three modalities into one RGB image? Measured on this data
(check_packing_vs_permodality.py), packing costs a LOT of fidelity, because three
decorrelated contrasts have to share a single 16-channel latent:
    T1w   30.57 -> 39.56 dB  (+8.99)
    T2w   33.24 -> 37.44 dB  (+4.20)
    FLAIR 31.92 -> 37.95 dB  (+6.03)
Per-modality also makes per-modality loss masking possible (training_plan.md 3/9).

Reads  <data_root>/metadata.jsonl + slice_*.npy  ([3,H,W] float [0,1])
Writes <out_root>/<same-relative-path>.npy       (latent [48,h,w] float16)
       <out_root>/metadata_latents.jsonl

Convention matches the trainer: latents = vae.encode(x).latent_dist.sample() * scaling_factor
(x scaled to [-1,1]). Run in a GPU container.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/rintern14/ymk/data_stage0_3t_reg")
    ap.add_argument("--out_root", default="/home/rintern14/ymk/data_stage0_3t_latents")
    ap.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae")
    ap.add_argument("--batch_size", type=int, default=16, help="slices per batch (x3 modalities on the GPU)")
    ap.add_argument("--packed", action="store_true",
                    help="legacy: encode the 3 modalities as one RGB image (16ch). Much worse fidelity.")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float32
    data_root, out_root = Path(args.data_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype).eval()
    scaling = vae.config.scaling_factor
    mode = "packed RGB (16ch)" if args.packed else "per-modality (48ch)"
    print(f"VAE latent_ch={vae.config.latent_channels} scaling={scaling} | mode={mode}")

    items = [json.loads(l) for l in open(data_root / "metadata.jsonl")]
    meta_out = open(out_root / "metadata_latents.jsonl", "w")

    buf_x, buf_it = [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).to(device, dtype) * 2.0 - 1.0   # [B,3,H,W] in [-1,1]
        B = x.shape[0]
        if args.packed:
            lat = vae.encode(x).latent_dist.sample() * scaling                # [B,16,h,w]
        else:
            # each modality -> its own 3-channel grayscale image -> its own 16ch latent
            g = x.permute(1, 0, 2, 3).reshape(B * 3, 1, *x.shape[-2:])        # [B*3,1,H,W] (mod-major)
            g = g.repeat(1, 3, 1, 1)                                          # [B*3,3,H,W]
            l = vae.encode(g).latent_dist.sample() * scaling                  # [B*3,16,h,w]
            h, w = l.shape[-2:]
            # back to [B, 3*16, h, w] ordered [T1_16 | T2_16 | FLAIR_16]
            lat = l.reshape(3, B, 16, h, w).permute(1, 0, 2, 3, 4).reshape(B, 48, h, w)
        lat = lat.to(torch.float16).cpu().numpy()
        for it, l_ in zip(buf_it, lat):
            rel = it["path"].rsplit(".", 1)[0] + ".npy"
            fp = out_root / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.save(fp, l_)
            meta_out.write(json.dumps({
                "path": rel, "dataset": it["dataset"], "subject": it["subject"],
                "slice": it["slice"], "target_avail": it["target_avail"],
            }) + "\n")
        buf_x.clear(); buf_it.clear()

    for i, it in enumerate(items):
        buf_x.append(np.load(data_root / it["path"]).astype("float32"))
        buf_it.append(it)
        if len(buf_x) >= args.batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{len(items)}", flush=True)
    flush()
    meta_out.close()
    print(f"DONE  {len(items)} latents -> {out_root}")


if __name__ == "__main__":
    main()
