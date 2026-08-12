"""Whole-volume PSNR/SSIM for Stage 1 checkpoints.

The in-training [held-out] metric samples only --eval_image_slices (16) central slices,
so it says little about the ends of each volume. This re-runs the same CFG sampler over
EVERY held-out slice (508) and reports, per checkpoint:

  psnr_vol    aggregate MSE over the whole volume -> one PSNR per (subject, modality)
  psnr_slice  mean of per-slice PSNRs (what the training log reports, on all slices)
  ssim        mean of per-slice SSIM

Reference images are decode48(target latent), exactly as in training, so the numbers are
comparable to the training log and share its VAE ceiling.

Results are appended to the output JSONL after each checkpoint, so a partial run is usable.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel  # noqa: E402
from train_stage1_cond import (AvailEmbedder, PairLatentDataset, add_condition_channels,  # noqa: E402
                               split_val_subjects, ssim)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", required=True, help="JSONL, one line per checkpoint")
    p.add_argument("--sd3", default=os.path.join(os.path.dirname(ROOT), "sd3_local"),
                   help="folder with scheduler/ and vae/")
    p.add_argument("--latent_root", default=os.path.join(os.path.dirname(ROOT), "data_stage1_latents"))
    p.add_argument("--null_embeds_path", default=None)
    p.add_argument("--text_embeds_path", default=None)
    p.add_argument("--val_subjects_per_ds", type=int, default=1)
    p.add_argument("--n_modalities", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--inference_shift", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=2.0)
    p.add_argument("--text_guidance_scale", type=float, default=1.5)
    p.add_argument("--no_text", action="store_true",
                   help="checkpoint was trained without --text_embeds_path: use the fixed NULL "
                        "embedding in every branch and drop the text CFG axis, like training did.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit_slices", type=int, default=0, help="debug: only the first N slices")
    a = p.parse_args()
    a.null_embeds_path = a.null_embeds_path or os.path.join(a.latent_root, "enhance_embeds.pt")
    a.text_embeds_path = a.text_embeds_path or os.path.join(a.latent_root, "source_embeds.pt")
    return a


def main():
    args = parse_args()
    dev = torch.device("cuda")
    dt = torch.bfloat16

    # ---- held-out set: same split as training (deterministic, seed 0) ----
    items = PairLatentDataset(args.latent_root).items
    val_keys = split_val_subjects(items, args.val_subjects_per_ds)
    heldout = PairLatentDataset(args.latent_root, split="val", val_keys=val_keys)
    n = len(heldout) if not args.limit_slices else min(args.limit_slices, len(heldout))
    print(f"hold-out {sorted(val_keys)} -> {n} slices", flush=True)

    data = {k: torch.stack([heldout[i][k] for i in range(n)])
            for k in ("input", "target", "input_avail", "target_avail")}
    meta = heldout.items[:n]
    src_names = [it["dataset"] for it in meta]

    # ---- conditioning banks ----
    blob = torch.load(args.null_embeds_path, map_location="cpu")
    null_prompt_embeds = blob["prompt_embeds"].to(dev, dt)
    null_pooled = blob["pooled_prompt_embeds"].to(dev, dt)
    tb = torch.load(args.text_embeds_path, map_location="cpu")
    names = sorted(tb)
    text_row = {nm: i for i, nm in enumerate(names)}
    null_row = text_row["null"]
    text_bank = torch.stack([tb[nm]["prompt_embeds"][0] for nm in names]).to(dev, dt)
    text_pooled = torch.stack([tb[nm]["pooled_prompt_embeds"][0] for nm in names]).to(dev, dt)

    vae = AutoencoderKL.from_pretrained(args.sd3, subfolder="vae").to(dev, dt).eval()
    vae.requires_grad_(False)
    base_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(args.sd3, subfolder="scheduler")

    target_ch = args.n_modalities * 16
    latent_hw = args.resolution // 8

    # One fixed noise draw shared by every checkpoint: otherwise the comparison measures
    # the draw as much as the weights.
    g = torch.Generator().manual_seed(args.seed)
    init_noise = torch.randn((n, target_ch, latent_hw, latent_hw), generator=g)

    def decode48(lat_scaled):
        lat = lat_scaled.to(dt) / vae.config.scaling_factor
        ch = lat.shape[1] // args.n_modalities
        chans = [vae.decode(lat[:, k * ch:(k + 1) * ch]).sample.mean(1)
                 for k in range(args.n_modalities)]
        return ((torch.stack(chans, 1) + 1) / 2).clamp(0, 1).float()

    @torch.no_grad()
    def sample_cond(net, iae, ae, cond, iav, tav, src, noise):
        """Same two-axis CFG as training's sample_cond, with a supplied initial latent."""
        b = cond.shape[0]
        sched = FlowMatchEulerDiscreteScheduler.from_config(
            base_sched.config, **({} if args.inference_shift is None else {"shift": args.inference_shift}))
        sched.set_timesteps(args.num_inference_steps, device=dev)
        lat = noise.to(dev, dt)
        zeros = torch.zeros_like(cond)
        if args.no_text:
            pe_null = pe_src = null_prompt_embeds.repeat(b, 1, 1)
            pooled_null = pooled_src = null_pooled.repeat(b, 1)
        else:
            rows = torch.tensor([text_row.get(d, null_row) for d in src], device=dev)
            pe_null = text_bank[torch.full((b,), null_row, device=dev)]
            pooled_null = text_pooled[torch.full((b,), null_row, device=dev)]
            pe_src, pooled_src = text_bank[rows], text_pooled[rows]

        t_emb = ae(tav)
        p_uncond = pooled_null + iae(torch.zeros_like(iav)) + t_emb
        p_img = pooled_null + iae(iav) + t_emb
        p_full = pooled_src + iae(iav) + t_emb
        s_img = args.guidance_scale
        s_txt = 0.0 if args.no_text else args.text_guidance_scale

        for t in sched.timesteps:
            tt = t.expand(b).to(dev)

            def run(x, emb, pooled):
                return net(hidden_states=x, timestep=tt, encoder_hidden_states=emb,
                           pooled_projections=pooled, return_dict=False)[0]

            with_img = torch.cat([lat, cond], 1)
            v = v_img = run(with_img, pe_null, p_img)
            if s_img != 1.0:
                v_uncond = run(torch.cat([lat, zeros], 1), pe_null, p_uncond)
                v = v_uncond + s_img * (v_img - v_uncond)
            if s_txt != 0.0:
                v = v + s_txt * (run(with_img, pe_src, p_full) - v_img)
            lat = sched.step(v, t, lat, return_dict=False)[0]
        return decode48(lat)

    @torch.no_grad()
    def evaluate(ckpt):
        t0 = time.time()
        net = SD3Transformer2DModel.from_pretrained(ckpt, subfolder="transformer")
        add_condition_channels(net, args.n_modalities * 16)      # no-op on a Stage-1 ckpt
        net = net.to(dev, dt).eval()
        net.requires_grad_(False)
        pooled_dim = null_pooled.shape[-1]
        iae, ae = AvailEmbedder(pooled_dim), AvailEmbedder(pooled_dim)
        iae.load_state_dict(torch.load(os.path.join(ckpt, "input_avail_embedder.pt"), map_location="cpu"))
        ae.load_state_dict(torch.load(os.path.join(ckpt, "avail_embedder.pt"), map_location="cpu"))
        iae, ae = iae.to(dev, dt).eval(), ae.to(dev, dt).eval()

        sse = defaultdict(float)     # (subject, modality) -> summed squared error
        npix = defaultdict(float)
        per_slice = []
        for s in range(0, n, args.batch_size):
            sl = slice(s, min(s + args.batch_size, n))
            cond = data["input"][sl].to(dev, dt)
            iav = data["input_avail"][sl].to(dev, dt)
            tav = data["target_avail"][sl].to(dev, dt)
            gen = sample_cond(net, iae, ae, cond, iav, tav, src_names[sl], init_noise[sl])
            ref = decode48(data["target"][sl].to(dev, dt))
            av = data["target_avail"][sl]
            for c in range(args.n_modalities):
                keep = av[:, c] > 0
                if not bool(keep.any()):
                    continue
                x, y = gen[keep, c], ref[keep, c]
                se = ((x - y) ** 2).flatten(1).sum(1)
                mse = se / (x.shape[-1] * x.shape[-2])
                psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-12))
                ss = ssim(x, y)
                idxs = [i for i, k in enumerate(keep.tolist()) if k]
                for j, i in enumerate(idxs):
                    it = meta[s + i]
                    key = (f"{it['dataset']}/{it['subject']}", c)
                    sse[key] += float(se[j])
                    npix[key] += x.shape[-1] * x.shape[-2]
                    per_slice.append({"subject": key[0], "slice": it["slice"], "modality": c,
                                      "psnr": round(float(psnr[j]), 4), "ssim": round(float(ss[j]), 5)})
            print(f"  {ckpt.rstrip('/').split('/')[-1]}: {min(s + args.batch_size, n)}/{n} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        pairs = []
        for key in sorted(sse):
            rows = [r for r in per_slice if r["subject"] == key[0] and r["modality"] == key[1]]
            pairs.append({
                "subject": key[0], "modality": key[1], "n_slices": len(rows),
                "psnr_vol": round(float(10 * np.log10(1.0 / max(sse[key] / npix[key], 1e-12))), 4),
                "psnr_slice": round(float(np.mean([r["psnr"] for r in rows])), 4),
                "ssim": round(float(np.mean([r["ssim"] for r in rows])), 5),
            })
        out = {
            "checkpoint": ckpt,
            "step": int(os.path.basename(ckpt.rstrip("/")).split("-")[-1]),
            "n_slices": n,
            "seconds": round(time.time() - t0, 1),
            "pairs": pairs,
            "overall": {
                "psnr_vol": round(float(np.mean([p["psnr_vol"] for p in pairs])), 4),
                "psnr_slice": round(float(np.mean([r["psnr"] for r in per_slice])), 4),
                "ssim": round(float(np.mean([r["ssim"] for r in per_slice])), 5),
            },
            "per_slice": per_slice,
        }
        del net, iae, ae
        torch.cuda.empty_cache()
        return out

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    for ckpt in args.checkpoints:
        res = evaluate(ckpt)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        o = res["overall"]
        print(f"[done] {ckpt}  psnr_vol={o['psnr_vol']:.3f}  psnr_slice={o['psnr_slice']:.3f}  "
              f"ssim={o['ssim']:.4f}  ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
