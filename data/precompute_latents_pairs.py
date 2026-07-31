"""
Stage 1 latent precompute: encode paired (ULF input, 3T target) slices.

Reads  <data_root>/metadata.jsonl + slice_*.npz  (keys: input[3,H,W], target[3,H,W])
Writes <out_root>/<same-rel>.npz                 (keys: input[48,h,w], target[48,h,w] fp16)
       <out_root>/metadata_latents.jsonl         (+ input_avail, target_avail)

Per-modality encoding (48ch), same convention as Stage 0's precompute_latents.py:
each modality replicated to 3ch -> VAE -> 16ch, concatenated [T1|T2|FLAIR].
Uses the LOCAL SD3 VAE -> no gated access. Run in a GPU container.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL


@torch.no_grad()
def encode_permod(vae, x, scaling):
    """x: [B,3,H,W] in [-1,1] -> [B,48,h,w] latent, ordered [T1_16|T2_16|FLAIR_16]."""
    B = x.shape[0]
    g = x.permute(1, 0, 2, 3).reshape(B * 3, 1, *x.shape[-2:]).repeat(1, 3, 1, 1)
    l = vae.encode(g).latent_dist.sample() * scaling
    h, w = l.shape[-2:]
    return l.reshape(3, B, 16, h, w).permute(1, 0, 2, 3, 4).reshape(B, 48, h, w)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/rintern14/ymk/data_stage1_pairs")
    ap.add_argument("--out_root", default="/home/rintern14/ymk/data_stage1_latents")
    ap.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda"
    data_root, out_root = Path(args.data_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, torch.float32).eval()
    scaling = vae.config.scaling_factor
    print(f"VAE latent_ch={vae.config.latent_channels} scaling={scaling}")

    items = [json.loads(l) for l in open(data_root / "metadata.jsonl")]
    meta_out = open(out_root / "metadata_latents.jsonl", "w")
    buf_in, buf_tg, buf_it = [], [], []

    def flush():
        if not buf_it:
            return
        xi = torch.from_numpy(np.stack(buf_in)).to(device, torch.float32) * 2 - 1
        xt = torch.from_numpy(np.stack(buf_tg)).to(device, torch.float32) * 2 - 1
        li = encode_permod(vae, xi, scaling).to(torch.float16).cpu().numpy()
        lt = encode_permod(vae, xt, scaling).to(torch.float16).cpu().numpy()
        for it, a, b in zip(buf_it, li, lt):
            rel = it["path"].rsplit(".", 1)[0] + ".npz"
            fp = out_root / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fp, input=a, target=b)
            meta_out.write(json.dumps({
                "path": rel, "dataset": it["dataset"], "subject": it["subject"], "slice": it["slice"],
                "input_avail": it["input_avail"], "target_avail": it["target_avail"]}) + "\n")
        buf_in.clear(); buf_tg.clear(); buf_it.clear()

    for i, it in enumerate(items):
        z = np.load(data_root / it["path"])
        buf_in.append(z["input"].astype("float32"))
        buf_tg.append(z["target"].astype("float32"))
        buf_it.append(it)
        if len(buf_it) >= args.batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"{i+1}/{len(items)}", flush=True)
    flush()
    meta_out.close()
    print(f"DONE {len(items)} pair-latents -> {out_root}")


if __name__ == "__main__":
    main()
