"""
Sample from a trained Stage 0 (unconditional 3T prior) checkpoint.

Standalone, so you can re-sample an existing checkpoint with different schedules
without retraining -- e.g. to see how much of the softness was the sampler rather
than the model:

    --inference_shift 3.0   SD3 default: only ~4 of 50 steps below sigma=0.25
    --inference_shift 1.0   ~12 of 50 steps there (detail-refining band)

Loads the per-modality 48ch transformer, decodes each modality's own 16ch block,
and writes a montage (rows = samples, cols = T1w | T2w | FLAIR).

Run in a GPU container.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel


class AvailEmbedder(nn.Module):
    """Must match the definition in train_stage0_uncond.py."""
    def __init__(self, pooled_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, pooled_dim))

    def forward(self, avail):
        return self.net(avail)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="dir containing transformer/ (e.g. stage0_uncond_v4/checkpoint-10000)")
    ap.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae")
    ap.add_argument("--null_embeds_path", default="/home/rintern14/ymk/data_stage0_3t_latents/null_embeds.pt")
    ap.add_argument("--scheduler_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside the checkpoint)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--inference_shift", type=float, default=None)
    ap.add_argument("--n_modalities", type=int, default=3)
    ap.add_argument("--avail", default="1,1,1", help="target_avail to condition on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(args.seed)

    transformer = SD3Transformer2DModel.from_pretrained(
        os.path.join(args.checkpoint, "transformer")).to(device, dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, dtype).eval()
    blob = torch.load(args.null_embeds_path, map_location="cpu")
    pe = blob["prompt_embeds"].to(device, dtype).repeat(args.n, 1, 1)
    pooled = blob["pooled_prompt_embeds"].to(device, dtype).repeat(args.n, 1)

    ae_path = os.path.join(args.checkpoint, "avail_embedder.pt")
    if os.path.exists(ae_path):
        ae = AvailEmbedder(pooled.shape[-1]).to(device, dtype)
        ae.load_state_dict(torch.load(ae_path, map_location="cpu"))
        ae.eval()
        av = torch.tensor([[float(v) for v in args.avail.split(",")]], device=device, dtype=dtype)
        pooled = pooled + ae(av.repeat(args.n, 1))
        print(f"avail conditioning: [{args.avail}]")
    else:
        print("no avail_embedder.pt -> unconditional")

    kw = {} if args.inference_shift is None else {"shift": args.inference_shift}
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler_path, subfolder="scheduler", **kw)
    sched.set_timesteps(args.num_inference_steps, device=device)
    sig = sched.sigmas
    print(f"shift={sched.config.shift}  steps={args.num_inference_steps}  "
          f"steps with sigma<0.25: {int((sig < 0.25).sum())}")

    ch = transformer.config.in_channels
    hw = args.resolution // 8
    lat = torch.randn((args.n, ch, hw, hw), device=device, dtype=dtype)
    for t in sched.timesteps:
        v = transformer(hidden_states=lat, timestep=t.expand(args.n).to(device),
                        encoder_hidden_states=pe, pooled_projections=pooled,
                        return_dict=False)[0]
        lat = sched.step(v, t, lat, return_dict=False)[0]

    lat = lat.to(dtype) / vae.config.scaling_factor
    if args.n_modalities > 1:
        c = lat.shape[1] // args.n_modalities
        img = torch.stack([vae.decode(lat[:, k * c:(k + 1) * c]).sample.mean(1)
                           for k in range(args.n_modalities)], dim=1)
    else:
        img = vae.decode(lat).sample
    img = ((img + 1) / 2).clamp(0, 1).float().cpu().numpy()

    grid = np.concatenate(
        [np.concatenate([img[i, c_] for c_ in range(3)], axis=1) for i in range(args.n)], axis=0)
    out = args.out or os.path.join(
        args.checkpoint,
        f"sample_shift{sched.config.shift}_steps{args.num_inference_steps}_seed{args.seed}.png")
    Image.fromarray((grid * 255).astype("uint8")).save(out)
    print(f"saved {out}  (rows=samples, cols=T1w|T2w|FLAIR)")


if __name__ == "__main__":
    main()
