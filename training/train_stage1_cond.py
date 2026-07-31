#!/usr/bin/env python
"""
Stage 0 -- Unconditional 3T MRI prior.

Fine-tunes a stock SD3 MMDiT (16-channel latent, NO channel expansion) on
[T1w, T2w, FLAIR] pseudo-RGB slices with a fixed null text prompt, using SD3's
rectified-flow / flow-matching objective. This learns p(3T) as required by
training_plan.md Section 4, and becomes the init for Stage 1 conditional.

Data: the <data_root>/metadata.jsonl + slice_*.npy produced by
data/register_and_preprocess_3t.py (channels = [T1w, T2w, FLAIR], float [0,1]).

Optional --use_avail injects target_avail ([1,1,0] for kcl, [1,1,1] else) into
the pooled projection via a zero-initialized MLP, so the model can tell a real
FLAIR from a copy-filled one. Zero-init => training starts as pure unconditional.

This file is self-contained (SD3 text-encoding helpers copied from the UltraEdit
trainer) so it does not import that module. Heavy job -> run in a GPU container.
"""
import argparse
import copy
import glob
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import (
    CLIPTokenizer, CLIPTextModelWithProjection,
    T5TokenizerFast, T5EncoderModel,
)
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.optimization import get_scheduler


# ----------------------------------------------------------------------- dataset
def split_val_subjects(items, n_val_per_ds, seed=0):
    """Pick n_val_per_ds hold-out SUBJECTS per dataset (not slices -- adjacent slices
    would leak between train/val). Returns the set of held-out 'dataset/subject' keys."""
    from collections import defaultdict
    by_ds = defaultdict(set)
    for it in items:
        by_ds[it["dataset"]].add(it["subject"])
    val = set()
    for ds, subs in by_ds.items():
        ordered = sorted(subs)
        # deterministic pseudo-shuffle without Math.random: rotate by a seed-derived offset
        off = seed % max(1, len(ordered))
        ordered = ordered[off:] + ordered[:off]
        for s in ordered[:n_val_per_ds]:
            val.add(f"{ds}/{s}")
    return val


class PairLatentDataset(Dataset):
    """Stage 1: precomputed paired latents. Returns input (ULF, 48ch) + target (3T, 48ch)
    + input_avail + target_avail. Produced by data/precompute_latents_pairs.py.

    split='train' excludes the hold-out subjects; split='val' keeps only them."""
    def __init__(self, data_root, datasets=None, split="all", val_keys=None):
        self.root = Path(data_root)
        with open(self.root / "metadata_latents.jsonl") as f:
            self.items = [json.loads(l) for l in f]
        if datasets:
            keep = set(datasets)
            self.items = [it for it in self.items if it["dataset"] in keep]
        if split != "all" and val_keys is not None:
            def is_val(it):
                return f"{it['dataset']}/{it['subject']}" in val_keys
            self.items = [it for it in self.items if (is_val(it) == (split == "val"))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        z = np.load(self.root / it["path"])
        return {
            "input": torch.from_numpy(z["input"].astype("float32")),     # [48,h,w]
            "target": torch.from_numpy(z["target"].astype("float32")),   # [48,h,w]
            "input_avail": torch.tensor(it["input_avail"], dtype=torch.float32),
            "target_avail": torch.tensor(it["target_avail"], dtype=torch.float32),
        }


class MRISliceDataset(Dataset):
    """(unused in Stage 1; kept for reference)"""
    def __init__(self, data_root, latent_mode=False, random_flip=True, datasets=None):
        self.root = Path(data_root)
        self.latent_mode = latent_mode
        meta = "metadata_latents.jsonl" if latent_mode else "metadata.jsonl"
        with open(self.root / meta) as f:
            self.items = [json.loads(l) for l in f]
        if datasets:
            keep = set(datasets)
            self.items = [it for it in self.items if it["dataset"] in keep]
        self.random_flip = random_flip and not latent_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.load(self.root / it["path"]).astype("float32")
        avail = torch.tensor(it["target_avail"], dtype=torch.float32)
        if self.latent_mode:
            return {"latents": torch.from_numpy(arr), "avail": avail}   # [16,h,w]
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, :, ::-1].copy()
        x = torch.from_numpy(arr) * 2.0 - 1.0                            # [3,H,W] -> [-1,1]
        return {"pixel_values": x, "avail": avail}


# ------------------------------------------------- SD3 text encoding (copied verbatim)
def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, dtype, prompt, device):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=max_sequence_length,
                            truncation=True, add_special_tokens=True, return_tensors="pt")
    emb = text_encoder(text_inputs.input_ids.to(device))[0]
    return emb.to(dtype=dtype, device=device)


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, dtype, device):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt")
    out = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    pooled = out[0]
    emb = out.hidden_states[-2].to(dtype=dtype, device=device)
    return emb, pooled


def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length, device, dtype):
    """Return (prompt_embeds [1, seq, 4096], pooled_prompt_embeds [1, 2048])."""
    clip_embeds, clip_pooled = [], []
    for tok, te in zip(tokenizers[:2], text_encoders[:2]):
        e, p = _encode_prompt_with_clip(te, tok, prompt, dtype, device)
        clip_embeds.append(e); clip_pooled.append(p)
    clip_prompt_embeds = torch.cat(clip_embeds, dim=-1)
    pooled = torch.cat(clip_pooled, dim=-1)
    t5_embed = _encode_prompt_with_t5(text_encoders[-1], tokenizers[-1],
                                      max_sequence_length, dtype, prompt, device)
    clip_prompt_embeds = F.pad(clip_prompt_embeds, (0, t5_embed.shape[-1] - clip_prompt_embeds.shape[-1]))
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_embed], dim=-2)
    return prompt_embeds, pooled


# ----------------------------------------------------------------- avail conditioning
def expand_transformer_channels(transformer, n_mod=3, base_in=16):
    """Widen SD3's 16-channel latent I/O to n_mod*16, for [T1_16 | T2_16 | FLAIR_16].

    Channels do NOT add tokens in a DiT, so attention cost is unchanged; only the
    patch-embed and the output head grow.

    input  (pos_embed.proj): replicate the pretrained weights across the modality
      blocks and divide by n_mod, so the initial response approximates the
      pretrained one on the mean modality.
    output (proj_out): replicate as-is, so every modality block starts out
      predicting what the pretrained model would have predicted.
    """
    # Decide against the STOCK SD3 width (base_in), not the current config: a resumed
    # checkpoint already reports in_channels=48, and deriving the target from that
    # would expand a second time (48 -> 144).
    old_in, new_in = base_in, base_in * n_mod
    cur = transformer.pos_embed.proj.in_channels
    if cur == new_in:
        return transformer                      # already expanded (resumed checkpoint)
    if cur != old_in:
        raise ValueError(f"unexpected patch-embed width {cur}: expected {old_in} (stock SD3) "
                         f"or {new_in} (already expanded)")
    patch = transformer.config.patch_size

    old_proj = transformer.pos_embed.proj                      # Conv2d(16, hidden, p, p)
    new_proj = nn.Conv2d(new_in, old_proj.out_channels, kernel_size=(patch, patch),
                         stride=patch, bias=old_proj.bias is not None)
    with torch.no_grad():
        # repeat tiles the 16-block n_mod times -> channel index k*16 + c
        new_proj.weight.copy_(old_proj.weight.data.repeat(1, n_mod, 1, 1) / n_mod)
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias.data)
    transformer.pos_embed.proj = new_proj.to(old_proj.weight.dtype)

    old_out = transformer.proj_out                             # Linear(inner, p*p*16)
    inner = old_out.in_features
    new_out = nn.Linear(inner, patch * patch * new_in, bias=old_out.bias is not None)
    with torch.no_grad():
        # unpatchify reads the last dim as (p, p, out_channels), channels innermost
        w = old_out.weight.data.reshape(patch * patch, old_in, inner)
        new_out.weight.copy_(w.repeat(1, n_mod, 1).reshape(patch * patch * new_in, inner))
        if old_out.bias is not None:
            b = old_out.bias.data.reshape(patch * patch, old_in)
            new_out.bias.copy_(b.repeat(1, n_mod).reshape(-1))
    transformer.proj_out = new_out.to(old_out.weight.dtype)

    transformer.register_to_config(in_channels=new_in, out_channels=new_in)
    transformer.out_channels = new_in                          # used by unpatchify
    print(f"expanded transformer latent I/O: {old_in} -> {new_in} channels")
    return transformer


def add_condition_channels(transformer, cond_ch):
    """Widen the input patch-embed to accept a concatenated condition (Stage 1).

    The Stage 0 checkpoint already has in==out==48 (per-modality target). Here we
    grow ONLY the input: pos_embed.proj 48 -> 48+cond_ch. The first 48 channels keep
    the loaded weights (noisy-target branch); the condition channels are ZERO-init,
    so at step 0 the condition has no effect and the model == the Stage 0 prior
    (training_plan.md §5). proj_out (output) stays 48 -- we still predict only 3T.
    """
    old = transformer.pos_embed.proj
    tgt_ch = old.in_channels
    if tgt_ch == 48 + cond_ch:
        return transformer                       # already widened (resumed)
    patch = transformer.config.patch_size
    new = nn.Conv2d(tgt_ch + cond_ch, old.out_channels, kernel_size=(patch, patch),
                    stride=patch, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :tgt_ch].copy_(old.weight.data)     # target branch <- Stage 0
        # condition branch (last cond_ch) stays zero
        if old.bias is not None:
            new.bias.copy_(old.bias.data)
    transformer.pos_embed.proj = new.to(old.weight.dtype)
    transformer.register_to_config(in_channels=tgt_ch + cond_ch)
    print(f"added condition channels: input {tgt_ch} -> {tgt_ch + cond_ch} "
          f"(target {tgt_ch} kept, condition {cond_ch} zero-init); output stays {tgt_ch}")
    return transformer


class AvailEmbedder(nn.Module):
    """MLP(avail[3]) -> pooled_dim, zero-initialized so it starts with no effect."""
    def __init__(self, pooled_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, pooled_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, avail):
        return self.net(avail)


# ------------------------------------------------------------------------------ train
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    p.add_argument("--data_root", default="/home/rintern14/ymk/data_stage0_3t_reg")
    p.add_argument("--latent_root", default=None,
                   help="if set, use precomputed VAE latents from here (no VAE loaded at train time)")
    p.add_argument("--null_embeds_path", default=None,
                   help="if set, load fixed null text embeds from here and never load text encoders (saves ~11GB)")
    p.add_argument("--output_dir", default="stage0_uncond_out")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=60000)
    p.add_argument("--learning_rate", type=float, default=5e-5, help="LR for the pretrained backbone")
    p.add_argument("--new_layer_lr_mult", type=float, default=10.0,
                   help="LR multiplier for the expanded I/O layers (pos_embed.proj, proj_out) "
                        "+ avail embedder, which are not genuinely pretrained")
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--weighting_scheme", default="logit_normal",
                   choices=["logit_normal", "sigma_sqrt", "cosmap"],
                   help="loss weighting. NOTE the loss is x0-parameterized, which is "
                        "equivalent to sigma^2 * velocity-MSE. 'logit_normal' -> weight 1 "
                        "(low-noise/detail band gets ~no gradient). 'sigma_sqrt' -> weight "
                        "sigma^-2, which cancels that and restores uniform velocity loss "
                        "(revives the detail band; better sharpness).")
    p.add_argument("--timestep_sampling", default="logit_normal", choices=["logit_normal", "uniform", "sigma_uniform"],
                   help="sigma_sqrt/cosmap weighting is meant to pair with uniform sampling; "
                        "logit_normal instead emphasizes mid-noise via sampling.")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--sigma_eps", type=float, default=1e-3,
                   help="floor on sigma when computing sigma^-2 weighting (numerical safety)")
    # periodic sampling
    p.add_argument("--validation_steps", type=int, default=1000, help="sample every N steps (0=off)")
    p.add_argument("--num_validation_samples", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--inference_shift", type=float, default=None,
                   help="override the scheduler's sigma shift at sampling time. SD3 ships "
                        "shift=3.0, which spends only ~4 of 50 steps below sigma=0.25 -- the "
                        "band that actually sharpens detail. 1.0 gives ~12 of 50 there.")
    p.add_argument("--vae_path", default="/home/rintern14/ymk/pretrained_models/dual_diff_sd3_512_base/vae",
                   help="VAE used to decode validation samples (must match the one that encoded the latents)")
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true",
                   help="8-bit AdamW (bitsandbytes): optimizer states 16GB->4GB, needed for ~44GB GPUs")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="restrict training to these source datasets (e.g. ulfenc) for a "
                        "single-source ablation. Default: all.")
    p.add_argument("--n_modalities", type=int, default=3,
                   help="latents are n_modalities x 16ch, ordered [T1|T2|FLAIR]. "
                        "1 = legacy packed-RGB 16ch latents.")
    p.add_argument("--mask_loss_by_avail", action="store_true",
                   help="drop missing target modalities from the loss (training_plan.md 9). "
                        "Requires per-modality latents (--n_modalities 3).")
    p.add_argument("--use_avail", action="store_true", help="inject target_avail into pooled projection")
    # ---- Stage 1 conditional ----
    p.add_argument("--stage0_checkpoint", default=None,
                   help="Stage 0 (48ch unconditional) checkpoint dir to initialize from. "
                        "The condition channels are added on top (zero-init).")
    p.add_argument("--flair_dropout_prob", type=float, default=0.2,
                   help="prob of hiding a present input FLAIR (§7 modality dropout), so the "
                        "model learns to synthesize it -- matches kcl's no-input-FLAIR case.")
    p.add_argument("--cond_dropout_prob", type=float, default=0.1,
                   help="prob of dropping the WHOLE condition (§6), enabling classifier-free guidance.")
    p.add_argument("--guidance_scale", type=float, default=2.0, help="CFG scale for validation sampling.")
    p.add_argument("--val_input_root", default=None,
                   help="input-only latents (unpaired ULF) for a generalization montage; "
                        "made by preprocess_val_inputs.py + precompute_latents.py")
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--checkpointing_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3345)
    p.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    p.add_argument("--tracker_project_name", default="ulfmri-stage1")
    p.add_argument("--run_name", default=None)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="path to a checkpoint-N dir: loads transformer/ + avail_embedder.pt, "
                        "resumes global_step and fast-forwards the LR schedule. "
                        "NOTE: optimizer state is not saved by this script, so momentum restarts.")
    return p.parse_args()


def main():
    args = parse_args()
    acc = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.report_to == "wandb" else None,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    if args.seed is not None:
        set_seed(args.seed)
    if acc.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    mp = args.pretrained_model_name_or_path
    use_null_file = args.null_embeds_path is not None
    if not use_null_file:
        tok1 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer_2")
        tok3 = T5TokenizerFast.from_pretrained(mp, subfolder="tokenizer_3")
        te1 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder")
        te2 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder_2")
        te3 = T5EncoderModel.from_pretrained(mp, subfolder="text_encoder_3")
    use_latents = args.latent_root is not None
    vae = None if use_latents else AutoencoderKL.from_pretrained(mp, subfolder="vae")
    # Stage 1 initializes from the Stage 0 (unconditional, 48ch) checkpoint, then adds
    # the condition channels. Resume takes precedence (already 48+48=96).
    init_ckpt = args.resume_from_checkpoint or args.stage0_checkpoint
    if init_ckpt:
        transformer = SD3Transformer2DModel.from_pretrained(init_ckpt, subfolder="transformer")
    else:
        transformer = SD3Transformer2DModel.from_pretrained(mp, subfolder="transformer")
        expand_transformer_channels(transformer, args.n_modalities)   # stock SD3 -> 48ch
    add_condition_channels(transformer, args.n_modalities * 16)       # 48 -> 96 (cond zero-init)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(mp, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    if vae is not None:
        vae.requires_grad_(False)
    transformer.requires_grad_(True)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Fixed NULL text embeddings. Preferred path: load precomputed embeds and never
    # bring the text encoders onto the GPU (T5-XXL alone is ~9.5GB). Fallback: load
    # the encoders, encode "", then FULLY free them (incl. the loop variable).
    if use_null_file:
        blob = torch.load(args.null_embeds_path, map_location="cpu")
        null_prompt_embeds = blob["prompt_embeds"].to(acc.device, dtype=weight_dtype)
        null_pooled = blob["pooled_prompt_embeds"].to(acc.device, dtype=weight_dtype)
    else:
        for _te in (te1, te2, te3):
            _te.requires_grad_(False)
            _te.to(acc.device, dtype=weight_dtype)
        with torch.no_grad():
            null_prompt_embeds, null_pooled = encode_prompt(
                [te1, te2, te3], [tok1, tok2, tok3], "",
                args.max_sequence_length, acc.device, weight_dtype)
        null_prompt_embeds = null_prompt_embeds.detach()
        null_pooled = null_pooled.detach()
        del te1, te2, te3, _te        # _te also pins the last encoder (T5) -> must delete
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 1 conditioning embedders (both zero-init): input_avail tells the model
    # which condition modalities are real; target_avail also masks the loss (§9).
    input_avail_embedder = AvailEmbedder(null_pooled.shape[-1])
    avail_embedder = AvailEmbedder(null_pooled.shape[-1]) if args.use_avail else None
    if args.resume_from_checkpoint:
        for name, mod in [("input_avail_embedder", input_avail_embedder), ("avail_embedder", avail_embedder)]:
            if mod is None:
                continue
            ap = os.path.join(args.resume_from_checkpoint, f"{name}.pt")
            if os.path.exists(ap):
                mod.load_state_dict(torch.load(ap, map_location="cpu"))
                if acc.is_main_process:
                    print(f"resumed {name} from {ap}", flush=True)

    if vae is not None:
        vae.to(acc.device, dtype=weight_dtype)
    transformer.to(acc.device)
    input_avail_embedder.to(acc.device)
    if avail_embedder is not None:
        avail_embedder.to(acc.device)

    # The expanded I/O layers are the only parts that are not genuinely pretrained
    # (they were replicated from 16ch weights), so they need a much larger LR than the
    # 24 pretrained MMDiT blocks, which a high LR would just damage.
    new_layer_names = ("pos_embed.proj", "proj_out")
    new_params, base_params = [], []
    for n, p in transformer.named_parameters():
        (new_params if any(n.startswith(k) or f".{k}" in n for k in new_layer_names)
         else base_params).append(p)
    new_params += list(input_avail_embedder.parameters())
    if avail_embedder is not None:
        new_params += list(avail_embedder.parameters())
    params = base_params + new_params
    groups = [
        {"params": base_params, "lr": args.learning_rate},
        {"params": new_params, "lr": args.learning_rate * args.new_layer_lr_mult},
    ]
    print(f"param groups: backbone={len(base_params)} tensors @ lr={args.learning_rate:g} | "
          f"new I/O={len(new_params)} tensors @ lr={args.learning_rate * args.new_layer_lr_mult:g}")
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    dataset = PairLatentDataset(args.latent_root, datasets=args.datasets)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                        num_workers=args.dataloader_num_workers, drop_last=True, pin_memory=True)

    lr_sched = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * acc.num_processes,
        num_training_steps=args.max_train_steps * acc.num_processes,
    )

    resume_step = 0
    resume_state_path = None
    if args.resume_from_checkpoint:
        resume_step = int(os.path.basename(os.path.normpath(args.resume_from_checkpoint)).split("-")[-1])
        cand = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(cand):
            # full resume: optimizer + LR scheduler are loaded after acc.prepare().
            resume_state_path = cand
        else:
            # weights-only resume (e.g. an old checkpoint without optimizer): fast-forward
            # the LR schedule so it picks up at the right point on the cosine curve.
            for _ in range(resume_step * acc.num_processes):
                lr_sched.step()
            if acc.is_main_process:
                print(f"resuming from step {resume_step} (weights only); LR schedule fast-forwarded", flush=True)

    transformer, optimizer, loader, lr_sched = acc.prepare(transformer, optimizer, loader, lr_sched)

    if resume_state_path:
        state = torch.load(resume_state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        lr_sched.load_state_dict(state["lr_scheduler"])
        if acc.is_main_process:
            print(f"resumed optimizer + LR scheduler from {resume_state_path} at step {resume_step}", flush=True)
    input_avail_embedder = acc.prepare(input_avail_embedder)
    if avail_embedder is not None:
        avail_embedder = acc.prepare(avail_embedder)

    def get_sigmas(timesteps, n_dim, dtype):
        sigmas = noise_scheduler_copy.sigmas.to(device=acc.device, dtype=dtype)
        schedule_t = noise_scheduler_copy.timesteps.to(acc.device)
        step_idx = [(schedule_t == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_idx].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---- periodic sampling: decode a few unconditional samples to eyeball progress ----
    val_vae = None
    if args.validation_steps and acc.is_main_process:
        val_vae = AutoencoderKL.from_pretrained(args.vae_path).to(acc.device, weight_dtype).eval()
        val_vae.requires_grad_(False)
    target_ch = args.n_modalities * 16        # we predict the 3T target (48ch)
    latent_hw = args.resolution // 8          # SD3 VAE downsamples 8x
    # fixed validation samples (real ULF input + its ground-truth 3T target)
    n_val = args.num_validation_samples
    _vi = [dataset[i] for i in range(min(n_val, len(dataset)))]
    val_inputs = torch.stack([d["input"] for d in _vi])
    val_targets = torch.stack([d["target"] for d in _vi])
    val_iavail = torch.stack([d["input_avail"] for d in _vi])

    def decode48(lat_scaled):
        """[n,48,h,w] scaled latent -> [n,3,H,W] numpy image in [0,1] (per-modality)."""
        lat = lat_scaled.to(weight_dtype) / val_vae.config.scaling_factor
        ch = lat.shape[1] // args.n_modalities
        chans = [val_vae.decode(lat[:, k * ch:(k + 1) * ch]).sample.mean(1)
                 for k in range(args.n_modalities)]
        img = torch.stack(chans, dim=1)
        return ((img + 1) / 2).clamp(0, 1).float().cpu().numpy()

    # optional unpaired validation inputs (real ULF, no 3T target) -> generalization check
    val_uinputs = val_uiavail = None
    if args.val_input_root:
        root = Path(args.val_input_root)
        uitems = [json.loads(l) for l in open(root / "metadata_latents.jsonl")]
        stride = max(1, len(uitems) // max(1, n_val))
        pick = uitems[::stride][:n_val]
        val_uinputs = torch.stack([torch.from_numpy(np.load(root / it["path"]).astype("float32")) for it in pick])
        val_uiavail = torch.stack([torch.tensor(it["target_avail"], dtype=torch.float32) for it in pick])

    def sample_cond(cond, iav):
        """CFG-sample 3T from a condition. cond/iav on device. Returns [n,3,H,W] numpy."""
        net = acc.unwrap_model(transformer)
        iae = acc.unwrap_model(input_avail_embedder)
        n = cond.shape[0]
        sched_kwargs = {} if args.inference_shift is None else {"shift": args.inference_shift}
        sched = FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config, **sched_kwargs)
        sched.set_timesteps(args.num_inference_steps, device=acc.device)
        lat = torch.randn((n, target_ch, latent_hw, latent_hw), device=acc.device, dtype=weight_dtype)
        zeros = torch.zeros_like(cond)
        pe = null_prompt_embeds.repeat(n, 1, 1)
        base = null_pooled.repeat(n, 1)
        pooled_c = base + iae(iav)
        pooled_u = base + iae(torch.zeros_like(iav))
        gs = args.guidance_scale
        for t in sched.timesteps:
            tt = t.expand(n).to(acc.device)
            v_c = net(hidden_states=torch.cat([lat, cond], 1), timestep=tt,
                      encoder_hidden_states=pe, pooled_projections=pooled_c, return_dict=False)[0]
            if gs != 1.0:
                v_u = net(hidden_states=torch.cat([lat, zeros], 1), timestep=tt,
                          encoder_hidden_states=pe, pooled_projections=pooled_u, return_dict=False)[0]
                v = v_u + gs * (v_c - v_u)
            else:
                v = v_c
            lat = sched.step(v, t, lat, return_dict=False)[0]
        return decode48(lat)

    def row(a):
        return np.concatenate([a[c] for c in range(3)], axis=1)

    def save_montage(rows_per_sample, name, step, caption):
        blocks = [np.pad(np.concatenate(rs, axis=0), ((0, 6), (0, 0))) for rs in rows_per_sample]
        grid = np.concatenate(blocks, axis=0)
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
        fp = os.path.join(args.output_dir, "samples", f"{name}_{step:06d}.png")
        Image.fromarray((grid * 255).astype("uint8")).save(fp)
        if args.report_to == "wandb":
            import wandb
            acc.log({name: wandb.Image(fp, caption=caption)}, step=step)
        print(f"validation -> {fp}", flush=True)

    @torch.no_grad()
    def run_validation(step):
        net = acc.unwrap_model(transformer)
        was_training = net.training
        net.eval()
        # Primary validation: unpaired ULF-only webb (no 3T target) -> 64mT input / output.
        # This is the real deployment case and uses subjects never seen in training.
        if val_uinputs is not None:
            uc = val_uinputs.to(acc.device, weight_dtype)
            ug = sample_cond(uc, val_uiavail.to(acc.device, weight_dtype))
            ui = decode48(uc)
            save_montage([[row(ui[i]), row(ug[i])] for i in range(uc.shape[0])],
                         "val", step,
                         f"step {step} — unpaired webb; rows: 64mT input / generated 3T; cols T1|T2|FLAIR")
        else:
            # fallback (no val_input_root): sample from training pairs, incl. real 3T
            cond = val_inputs.to(acc.device, weight_dtype)
            gen = sample_cond(cond, val_iavail.to(acc.device, weight_dtype))
            inp = decode48(cond)
            tgt = decode48(val_targets.to(acc.device, weight_dtype))
            save_montage([[row(inp[i]), row(gen[i]), row(tgt[i])] for i in range(cond.shape[0])],
                         "val", step,
                         f"step {step} — rows: 64mT input / generated 3T / real 3T; cols T1|T2|FLAIR")
        if was_training:
            net.train()

    if args.report_to == "wandb" and acc.is_main_process:
        acc.init_trackers(
            args.tracker_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    if acc.is_main_process:
        print(f"dataset slices={len(dataset)}  steps={args.max_train_steps}  use_avail={args.use_avail}")

    global_step = resume_step
    done = False
    while not done:
        for batch in loader:
            with acc.accumulate(transformer):
                latents = batch["target"].to(acc.device, dtype=weight_dtype)   # 3T target 48ch
                cond = batch["input"].to(acc.device, dtype=weight_dtype)       # ULF input 48ch
                in_avail = batch["input_avail"].to(acc.device)                 # [B,3]
                bsz = latents.shape[0]
                ch_per_mod = latents.shape[1] // args.n_modalities

                # --- condition-side modality dropout (§7): on inputs that HAVE a modality,
                #     randomly hide it (zero its latent block + mark input_avail=0) so the
                #     model learns to fill it in. We only drop FLAIR (the real deployment
                #     case: kcl has no input FLAIR). Target stays full.
                if args.flair_dropout_prob > 0:
                    drop = (torch.rand(bsz, device=acc.device) < args.flair_dropout_prob) & (in_avail[:, 2] > 0)
                    if drop.any():
                        fl = slice(2 * ch_per_mod, 3 * ch_per_mod)
                        cond[drop, fl] = 0.0
                        in_avail = in_avail.clone(); in_avail[drop, 2] = 0.0

                # --- classifier-free guidance dropout (§6): drop the WHOLE condition so the
                #     model also learns v(x|no condition); enables guidance at inference.
                if args.cond_dropout_prob > 0:
                    cfg_drop = (torch.rand(bsz, device=acc.device) < args.cond_dropout_prob)
                    if cfg_drop.any():
                        cond[cfg_drop] = 0.0
                        in_avail = in_avail.clone(); in_avail[cfg_drop] = 0.0

                noise = torch.randn_like(latents)
                if args.timestep_sampling == "sigma_uniform":
                    # Uniform over NOISE LEVEL, not over schedule index. With SD3's
                    # shift=3.0 the index grid is heavily skewed to high sigma
                    # (sigma<0.25 is only the top ~10% of indices), so index-uniform
                    # sampling starves the low-sigma band where fine detail is learned.
                    tgt = torch.rand(size=(bsz,))
                    all_sig = noise_scheduler_copy.sigmas.cpu()
                    idx = (all_sig[None, :] - tgt[:, None]).abs().argmin(dim=1)
                else:
                    if args.timestep_sampling == "logit_normal":
                        u = torch.sigmoid(torch.normal(args.logit_mean, args.logit_std,
                                                       size=(bsz,), device="cpu"))
                    else:
                        u = torch.rand(size=(bsz,), device="cpu")
                    idx = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                idx = idx.clamp(0, noise_scheduler_copy.timesteps.numel() - 1)
                timesteps = noise_scheduler_copy.timesteps[idx].to(acc.device)
                sigmas = get_sigmas(timesteps, latents.ndim, latents.dtype)
                noisy = sigmas * noise + (1.0 - sigmas) * latents

                pe = null_prompt_embeds.repeat(bsz, 1, 1)                 # fixed "enhance" text
                pooled = null_pooled.repeat(bsz, 1)
                pooled = pooled + input_avail_embedder(in_avail.to(weight_dtype))
                if avail_embedder is not None:
                    pooled = pooled + avail_embedder(batch["target_avail"].to(acc.device, dtype=weight_dtype))

                model_input = torch.cat([noisy, cond], dim=1)             # 48 + 48 = 96ch
                model_pred = transformer(
                    hidden_states=model_input, timestep=timesteps,
                    encoder_hidden_states=pe, pooled_projections=pooled,
                    return_dict=False,
                )[0]
                # sigma^-2 weighting on the x0-MSE is algebraically identical to a plain
                # velocity MSE (x0_pred - x0 = sigma*(v - v_pred)). Computing it directly
                # in velocity space avoids forming a tiny sigma*dv in bf16 and then blowing
                # it up by sigma^-2, which destroys precision at small sigma.
                if args.weighting_scheme == "sigma_sqrt":
                    target = noise
                    model_pred = model_pred + latents      # v_pred + x0 == noise_pred
                else:
                    model_pred = model_pred * (-sigmas) + noisy   # -> x0 prediction
                    target = latents
                # x0-MSE == sigma^2 * velocity-MSE, so weighting decides which noise band
                # actually gets gradient. sigma^-2 cancels the sigma^2 and restores a
                # uniform velocity loss -> the low-noise (fine detail) band is revived.
                if args.weighting_scheme == "sigma_sqrt":
                    # already absorbed above by comparing in velocity space -- applying
                    # sigma^-2 again here would double-count it.
                    weighting = torch.ones_like(sigmas)
                elif args.weighting_scheme == "cosmap":
                    bot = 1 - 2 * sigmas + 2 * sigmas ** 2
                    weighting = 2 / (math.pi * bot)
                else:                                        # logit_normal -> uniform weight
                    weighting = torch.ones_like(sigmas)
                sq = weighting.float() * (model_pred.float() - target.float()) ** 2
                if args.mask_loss_by_avail and args.n_modalities > 1:
                    # target_avail [B,3] -> per-latent-channel mask [B,48,1,1]; a missing
                    # modality (KCL has no FLAIR) contributes no gradient, instead of being
                    # supervised against a copy-filled stand-in.
                    m = batch["target_avail"].to(acc.device).float()
                    m = m.repeat_interleave(ch_per_mod, dim=1)[:, :, None, None]
                    loss = (sq * m).sum() / (m.sum() * target.shape[-2] * target.shape[-1] + 1e-8)
                else:
                    loss = sq.reshape(bsz, -1).mean(1).mean()

                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if acc.sync_gradients:
                global_step += 1
                if args.report_to == "wandb":
                    acc.log({"train_loss": loss.item(), "lr": lr_sched.get_last_lr()[0]}, step=global_step)
                if acc.is_main_process and global_step % args.log_every == 0:
                    print(f"step {global_step}/{args.max_train_steps}  loss {loss.item():.4f}", flush=True)
                if (args.validation_steps and acc.is_main_process
                        and global_step % args.validation_steps == 0):
                    run_validation(global_step)
                if global_step % args.checkpointing_steps == 0 and acc.is_main_process:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    acc.unwrap_model(transformer).save_pretrained(os.path.join(save_dir, "transformer"))
                    torch.save(acc.unwrap_model(input_avail_embedder).state_dict(),
                               os.path.join(save_dir, "input_avail_embedder.pt"))
                    if avail_embedder is not None:
                        torch.save(acc.unwrap_model(avail_embedder).state_dict(),
                                   os.path.join(save_dir, "avail_embedder.pt"))
                    # Full training state (optimizer + LR scheduler) for exact resume.
                    # It's large (~optimizer size), so keep it only on the latest checkpoint
                    # and prune it from all earlier ones.
                    torch.save({"optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_sched.state_dict(),
                                "global_step": global_step},
                               os.path.join(save_dir, "training_state.pt"))
                    for old in glob.glob(os.path.join(args.output_dir, "checkpoint-*", "training_state.pt")):
                        if os.path.abspath(old) != os.path.abspath(os.path.join(save_dir, "training_state.pt")):
                            os.remove(old)
                    print(f"saved {save_dir} (+ optimizer state)", flush=True)
                if global_step >= args.max_train_steps:
                    done = True
                    break

    if acc.is_main_process:
        acc.unwrap_model(transformer).save_pretrained(os.path.join(args.output_dir, "transformer"))
        print("training done")
    if args.report_to == "wandb":
        acc.end_training()


if __name__ == "__main__":
    main()
