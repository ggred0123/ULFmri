"""Stage 1 inference: 64mT -> 3T.

Consumes the preprocessed slice format the data scripts already produce
(<input_root>/metadata.jsonl + slice_*.npy|npz with a [3,256,256] float image in
[0,1], channels [T1,T2,FLAIR]) -- use data/preprocess_val_inputs.py to get there
from NIfTI. Encodes to the 48ch latent, samples with two-axis CFG, decodes, and
writes one .npy plus a PNG montage per slice.

Defaults are the settings that won the held-out sweep on stage1_cond_v3_text:
  guidance_scale 2.0, text_guidance_scale 1.0, inference_shift 1.0, 50 steps.
--num_samples > 1 averages independent draws: diffusion returns a sample, not the
posterior mean, and averaging 4 measured +1.2dB PSNR for 4x the compute.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

MODS = ["T1w", "T2w", "FLAIR"]


class AvailEmbedder(nn.Module):
    """Must match training/train_stage1_cond.py."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, d))

    def forward(self, a):
        return self.net(a)


def load_vae(path):
    sub = None if (Path(path) / "config.json").exists() else "vae"
    return AutoencoderKL.from_pretrained(path, subfolder=sub)


def load_slice(p):
    """Preprocessed slice -> [3,H,W] float32. Pairs are .npz (input/target), val is .npy."""
    if p.suffix == ".npz":
        z = np.load(p)
        return z["input" if "input" in z else "target"].astype("float32")
    return np.load(p).astype("float32")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a checkpoint-N dir (transformer/ + *_avail_embedder.pt)")
    ap.add_argument("--input_root", required=True, help="dir with metadata.jsonl + preprocessed slices")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--vae_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--null_embeds_path", required=True, help="fixed prompt embeds (enhance_embeds.pt)")
    ap.add_argument("--text_embeds_path", default=None, help="per-source embeds; needed for text CFG")
    ap.add_argument("--source", default=None, help="prompt name to use (default: each slice's own dataset)")
    ap.add_argument("--guidance_scale", type=float, default=2.0, help="image CFG")
    ap.add_argument("--text_guidance_scale", type=float, default=1.0, help="text CFG (0 = off, 2 passes)")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--inference_shift", type=float, default=1.0)
    ap.add_argument("--num_samples", type=int, default=1, help="average N draws (higher fidelity, N x cost)")
    ap.add_argument("--target_avail", default="1,1,1", help="which 3T modalities to ask for")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_png", action="store_true")
    args = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    root, out = Path(args.input_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(root / "metadata.jsonl")]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} slices from {root}", flush=True)

    vae = load_vae(args.vae_path).to(dev, dt).eval()
    net = SD3Transformer2DModel.from_pretrained(args.checkpoint, subfolder="transformer").to(dev, dt).eval()
    sched_cfg = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler").config

    blob = torch.load(args.null_embeds_path, map_location="cpu")
    pe_fixed = blob["prompt_embeds"].to(dev, dt)
    po_fixed = blob["pooled_prompt_embeds"].to(dev, dt)

    bank = None
    if args.text_embeds_path:
        tb = torch.load(args.text_embeds_path, map_location="cpu")
        names = sorted(tb)
        bank = {"row": {n: i for i, n in enumerate(names)},
                "pe": torch.stack([tb[n]["prompt_embeds"][0] for n in names]).to(dev, dt),
                "po": torch.stack([tb[n]["pooled_prompt_embeds"][0] for n in names]).to(dev, dt)}
        print(f"text prompts: {names}", flush=True)
    use_text = bank is not None and args.text_guidance_scale != 0.0
    if args.text_guidance_scale != 0.0 and bank is None:
        print("WARNING: --text_guidance_scale set but no --text_embeds_path; text CFG disabled", flush=True)

    def emb(name):
        m = AvailEmbedder(po_fixed.shape[-1])
        m.load_state_dict(torch.load(Path(args.checkpoint) / f"{name}.pt", map_location="cpu"))
        return m.to(dev, dt).eval()

    iae, ae = emb("input_avail_embedder"), emb("avail_embedder")
    want = torch.tensor([float(x) for x in args.target_avail.split(",")], device=dev, dtype=dt)

    def encode48(x):
        """[B,3,H,W] in [0,1] -> [B,48,h,w] latent, per-modality, matching precompute_latents_pairs."""
        x = x * 2 - 1
        b = x.shape[0]
        g = x.permute(1, 0, 2, 3).reshape(b * 3, 1, *x.shape[-2:]).repeat(1, 3, 1, 1)
        l = vae.encode(g).latent_dist.sample() * vae.config.scaling_factor
        h, w = l.shape[-2:]
        return l.reshape(3, b, 16, h, w).permute(1, 0, 2, 3, 4).reshape(b, 48, h, w)

    def decode48(l):
        l = l.to(dt) / vae.config.scaling_factor
        ch = l.shape[1] // 3
        img = torch.stack([vae.decode(l[:, k * ch:(k + 1) * ch]).sample.mean(1) for k in range(3)], 1)
        return ((img + 1) / 2).clamp(0, 1).float()

    def sample(cond, iav, rows_src, seed):
        n = cond.shape[0]
        sch = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg, shift=args.inference_shift)
        sch.set_timesteps(args.num_inference_steps, device=dev)
        g = torch.Generator(device=dev).manual_seed(seed)
        lat = torch.randn((n, 48, cond.shape[-2], cond.shape[-1]), device=dev, dtype=dt, generator=g)
        if use_text:
            r_null = torch.full((n,), bank["row"]["null"], device=dev)
            pe_n, po_n = bank["pe"][r_null], bank["po"][r_null]
            pe_s, po_s = bank["pe"][rows_src], bank["po"][rows_src]
        else:
            pe_n = pe_s = pe_fixed.repeat(n, 1, 1)
            po_n = po_s = po_fixed.repeat(n, 1)
        te = ae(want.expand(n, 3))
        p_un = po_n + iae(torch.zeros_like(iav)) + te
        p_im = po_n + iae(iav) + te
        p_fu = po_s + iae(iav) + te
        zeros = torch.zeros_like(cond)
        for t in sch.timesteps:
            tt = t.expand(n).to(dev)

            def f(x, e, p):
                return net(hidden_states=x, timestep=tt, encoder_hidden_states=e,
                           pooled_projections=p, return_dict=False)[0]

            wi = torch.cat([lat, cond], 1)
            v = v_img = f(wi, pe_n, p_im)
            if args.guidance_scale != 1.0:
                v_un = f(torch.cat([lat, zeros], 1), pe_n, p_un)
                v = v_un + args.guidance_scale * (v_img - v_un)
            if use_text:
                v = v + args.text_guidance_scale * (f(wi, pe_s, p_fu) - v_img)
            lat = sch.step(v, t, lat, return_dict=False)[0]
        return decode48(lat)

    meta_f = open(out / "metadata.jsonl", "w")
    per_sub = defaultdict(int)
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        x = torch.from_numpy(np.stack([load_slice(root / it["path"]) for it in chunk])).to(dev, dt)
        cond = encode48(x)
        av = [it.get("input_avail", it.get("target_avail", [1, 1, 1])) for it in chunk]
        iav = torch.tensor(av, dtype=torch.float32, device=dev).to(dt)
        rows = None
        if bank is not None:
            src = [args.source or it.get("dataset", "null") for it in chunk]
            rows = torch.tensor([bank["row"].get(s, bank["row"]["null"]) for s in src], device=dev)

        acc = None
        for k in range(args.num_samples):
            g = sample(cond, iav, rows, args.seed + k)
            acc = g if acc is None else acc + g
        gen = (acc / args.num_samples).cpu().numpy()

        for it, out_img, in_img in zip(chunk, gen, x.float().cpu().numpy()):
            sub = it.get("subject", "sub")
            d = out / it.get("dataset", "ds") / sub
            d.mkdir(parents=True, exist_ok=True)
            stem = Path(it["path"]).stem
            np.save(d / f"{stem}.npy", out_img.astype("float32"))
            if not args.no_png:
                strip = np.concatenate([np.concatenate([in_img[c] for c in range(3)], 1),
                                        np.concatenate([out_img[c] for c in range(3)], 1)], 0)
                Image.fromarray((strip * 255).astype("uint8")).save(d / f"{stem}.png")
            meta_f.write(json.dumps({"path": str((d / f"{stem}.npy").relative_to(out)),
                                     "dataset": it.get("dataset"), "subject": sub,
                                     "slice": it.get("slice"), "target_avail": want.tolist()}) + "\n")
            per_sub[sub] += 1
        print(f"  {min(i + args.batch_size, len(items))}/{len(items)}", flush=True)

    meta_f.close()
    print(f"\nDONE -> {out}  ({dict(per_sub)})")
    print(f"  s_img={args.guidance_scale} s_txt={args.text_guidance_scale if use_text else 0.0} "
          f"steps={args.num_inference_steps} shift={args.inference_shift} avg={args.num_samples}")
    if not args.no_png:
        print("  PNG rows: 64mT input / generated 3T, cols T1|T2|FLAIR")


if __name__ == "__main__":
    main()
