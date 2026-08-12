#!/usr/bin/env python
"""
Stage 0 -- Unconditional 3T MRI prior.

Fine-tunes a stock SD3 MMDiT (16-channel latent, NO channel expansion) on
[T1w, T2w, FLAIR] pseudo-RGB slices, using SD3's rectified-flow / flow-matching
objective. This learns p(3T) as required by training_plan.md Section 4, and
becomes the init for Stage 1 conditional.

Data: the <data_root>/metadata.jsonl + <orientation>_<kkk>.npy produced by
data/register_and_preprocess_3t.py (channels = [T1w, T2w, FLAIR], float [0,1]),
cut in all three planes -- axial, coronal, sagittal.

Text: with --text_embeds_path the slice orientation IS the prompt (one embedding
per plane from data/precompute_text_embeds.py, "" as the dropout / CFG-negative
branch), so the prior is p(3T | plane) and sampling can ask for a specific plane.
Without it the run falls back to the old fixed-null-prompt behaviour.

Optional --use_avail injects target_avail ([1,1,0] for kcl, [1,1,1] else) into
the pooled projection via a zero-initialized MLP, so the model can tell a real
FLAIR from a copy-filled one. Zero-init => training starts as pure unconditional.

This file is self-contained (SD3 text-encoding helpers copied from the UltraEdit
trainer) so it does not import that module. Heavy job -> run in a GPU container.
"""
import argparse
import collections
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


# --------------------------------------------------- acquisition-plane validity
# Which 3T volumes were acquired as 2D slice stacks rather than 3D. A 2D scan only
# has real resolution inside the plane it was acquired in, so its coronal and
# sagittal reslices are a stack of smeared slabs -- see qc_stage0_3ori/. Confirmed
# from the archived pre-resample headers for webb and kcl
# (medical/*/derivatives/*/reference/processing/), and measured for ulfenc, whose
# originals were not archived:
#
#            T1w             T2w                FLAIR
#   kcl      3D 1.0mm        3D 1.0mm           (none)
#   ulfenc   3D              2D axial ~5.2mm    3D
#   webb     3D 1.0mm        2D axial 3.0mm     2D axial 5.5mm
#
# Loss weight, NOT target_avail: target_avail means "this slot is a copy-fill, not a
# real observation" and is also fed to the model as conditioning, so overloading it
# with "real but resolution-invalid here" would teach the embedder two different
# things under one code. This only ever touches the loss.
SLAB_ACQUIRED = {("ulfenc", "T2w"), ("webb", "T2w"), ("webb", "FLAIR")}
MODALITIES = ["T1w", "T2w", "FLAIR"]


def resolution_weight(dataset, orientation, out_of_plane_weight):
    """Per-modality loss weight for one slice: 1.0 in the acquired plane, and for
    everything acquired in 3D; out_of_plane_weight where a 2D scan is being viewed
    across its slabs."""
    if orientation == "axial":                      # every 2D scan here is axial
        return [1.0] * 3
    return [out_of_plane_weight if (dataset, m) in SLAB_ACQUIRED else 1.0 for m in MODALITIES]


# ----------------------------------------------------------------------- dataset
class MRISliceDataset(Dataset):
    """image mode: returns [3,H,W] pixels in [-1,1] (VAE-encoded in the loop).
    latent mode: returns precomputed [16,h,w] latents (no VAE at train time)."""
    def __init__(self, data_root, latent_mode=False, random_flip=True, datasets=None,
                 axial_only=(), out_of_plane_weight=1.0):
        self.root = Path(data_root)
        self.latent_mode = latent_mode
        self.out_of_plane_weight = out_of_plane_weight
        meta = "metadata_latents.jsonl" if latent_mode else "metadata.jsonl"
        with open(self.root / meta) as f:
            self.items = [json.loads(l) for l in f]
        if datasets:                     # e.g. ["ulfenc"] for a single-source ablation
            keep = set(datasets)
            self.items = [it for it in self.items if it["dataset"] in keep]
        if axial_only:                   # sites whose out-of-plane slices are not worth keeping
            drop = set(axial_only)
            self.items = [it for it in self.items
                          if it["dataset"] not in drop or it.get("orientation", "axial") == "axial"]
        # flipping a latent spatially is not a true image flip -> only augment in image mode
        self.random_flip = random_flip and not latent_mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.load(self.root / it["path"]).astype("float32")
        avail = torch.tensor(it["target_avail"], dtype=torch.float32)
        # picks this sample's prompt row (default collate keeps it a list of str).
        # Datasets built before the 3-plane rewrite are axial-only and say nothing.
        ori = it.get("orientation", "axial")
        res_w = torch.tensor(resolution_weight(it["dataset"], ori, self.out_of_plane_weight),
                             dtype=torch.float32)
        if self.latent_mode:
            return {"latents": torch.from_numpy(arr), "avail": avail, "orientation": ori,
                    "res_weight": res_w, "dataset": it["dataset"]}
        # Only the left-right mirror is an anatomically valid augmentation, and which
        # array axis that is depends on the plane: axial is [L->R, P->A] so it is the
        # ROW axis, coronal is [S->I, L->R] so it is the column axis, and a sagittal
        # slice has no in-plane mirror at all (flipping it would swap front and back).
        flip_axis = {"axial": -2, "coronal": -1}.get(ori)
        if self.random_flip and flip_axis is not None and random.random() < 0.5:
            arr = np.flip(arr, axis=flip_axis).copy()
        x = torch.from_numpy(arr) * 2.0 - 1.0                            # [3,H,W] -> [-1,1]
        return {"pixel_values": x, "avail": avail, "orientation": ori,
                "res_weight": res_w, "dataset": it["dataset"]}


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
    p.add_argument("--text_embeds_path", default=None,
                   help="bank of per-orientation prompt embeddings from "
                        "data/precompute_text_embeds.py --scheme orientation. Each slice is "
                        "conditioned on its own plane; without this every sample gets the "
                        "same null prompt and the text path carries no information.")
    p.add_argument("--text_dropout_prob", type=float, default=0.1,
                   help="probability of swapping a sample's prompt for the null ('') one, so "
                        "the unconditional branch stays trained and text CFG is well defined")
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
    p.add_argument("--cond_lr_mult", type=float, default=1.0,
                   help="LR multiplier for the conditioning path (time_text_embed + the "
                        "avail embedder feeding it). Its input is nearly constant, so its "
                        "gradient never averages out and it drifts until it destabilises the "
                        "AdaLN modulation of every block -- it owned 82-85%% of the gradient "
                        "at every spike. 0.05-0.1 keeps it trainable without the runaway; "
                        "0 freezes it at the pretrained projection.")
    p.add_argument("--adam_beta2", type=float, default=0.999,
                   help="Adam second-moment decay. On a weights-only resume the moments "
                        "restart at zero and 0.999 needs ~1000 steps to become reliable -- "
                        "exactly the horizon at which the v2 and v3 resumes both blew up. "
                        "0.95 adapts in ~20 steps instead.")
    p.add_argument("--grad_spike_abs", type=float, default=0.0,
                   help="hard ceiling on the pre-clip gradient norm; a step above it is "
                        "refused outright. Backstop for the relative rule, whose EMA can be "
                        "dragged upward by a long spike storm (it was, in v3).")
    p.add_argument("--resume_fresh_schedule", action="store_true",
                   help="on resume, run the LR schedule from zero over the REMAINING steps "
                        "(warmup included) instead of fast-forwarding into the middle of the "
                        "cosine. Pairs with the moment restart: full LR on cold moments is "
                        "what makes a resume diverge a few hundred steps later.")
    p.add_argument("--grad_spike_norm", type=float, default=10.0,
                   help="refuse the optimizer step when the pre-clip gradient norm exceeds "
                        "this multiple of its running EMA (and is above 2.0). This is the "
                        "guard that matters: the gradient blows up well before the loss does. "
                        "0 disables.")
    p.add_argument("--loss_spike_factor", type=float, default=8.0,
                   help="skip a batch whose loss exceeds this multiple of the running loss "
                        "EMA, so one bad batch cannot poison the Adam moments. 0 disables. "
                        "The v1 run diverged at step ~20k and never recovered without this.")
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
    p.add_argument("--axial_only", nargs="+", default=(),
                   help="sites to keep axial slices from only. webb belongs here: its 3T T2w "
                        "(3.0mm) and FLAIR (5.5mm) are 2D axial, so two of its three channels "
                        "are unusable out of plane and only 10 subjects are on offer.")
    p.add_argument("--out_of_plane_weight", type=float, default=1.0,
                   help="loss weight for a modality viewed across the slabs of a 2D "
                        "acquisition (see SLAB_ACQUIRED). 1.0 = no correction, 0.0 = drop it "
                        "from the loss entirely. Applied to the loss only, never to the "
                        "target_avail the model is conditioned on.")
    p.add_argument("--n_modalities", type=int, default=3,
                   help="latents are n_modalities x 16ch, ordered [T1|T2|FLAIR]. "
                        "1 = legacy packed-RGB 16ch latents.")
    p.add_argument("--mask_loss_by_avail", action="store_true",
                   help="drop missing target modalities from the loss (training_plan.md 9). "
                        "Requires per-modality latents (--n_modalities 3).")
    p.add_argument("--use_avail", action="store_true", help="inject target_avail into pooled projection")
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--checkpointing_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3345)
    p.add_argument("--report_to", default="wandb", choices=["wandb", "none"])
    p.add_argument("--tracker_project_name", default="ulfmri-stage0")
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
    # A prompt bank already contains its "" row, so it also removes any reason to
    # bring the ~11GB of text encoders onto the GPU just to encode the null prompt.
    use_null_file = args.null_embeds_path is not None or args.text_embeds_path is not None
    if not use_null_file:
        tok1 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer")
        tok2 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer_2")
        tok3 = T5TokenizerFast.from_pretrained(mp, subfolder="tokenizer_3")
        te1 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder")
        te2 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder_2")
        te3 = T5EncoderModel.from_pretrained(mp, subfolder="text_encoder_3")
    use_latents = args.latent_root is not None
    vae = None if use_latents else AutoencoderKL.from_pretrained(mp, subfolder="vae")
    if args.resume_from_checkpoint:
        transformer = SD3Transformer2DModel.from_pretrained(args.resume_from_checkpoint, subfolder="transformer")
    else:
        transformer = SD3Transformer2DModel.from_pretrained(mp, subfolder="transformer")
    if args.n_modalities > 1:
        expand_transformer_channels(transformer, args.n_modalities)
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
        blob = torch.load(args.null_embeds_path or args.text_embeds_path, map_location="cpu")
        if args.null_embeds_path is None:
            blob = blob["null"]                     # prompt bank -> its "" row
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

    # Per-orientation prompts (optional). Stacked into a bank so a batch is one
    # index_select instead of a dict lookup per sample. Row `null_row` is the ""
    # prompt: the text-dropout target during training and the CFG-negative branch.
    text_bank = text_pooled_bank = text_row = None
    null_row = 0
    if args.text_embeds_path:
        tb = torch.load(args.text_embeds_path, map_location="cpu")
        names = sorted(tb)
        text_row = {n: i for i, n in enumerate(names)}
        null_row = text_row["null"]
        text_bank = torch.stack([tb[n]["prompt_embeds"][0] for n in names]).to(acc.device, weight_dtype)
        text_pooled_bank = torch.stack([tb[n]["pooled_prompt_embeds"][0] for n in names]).to(acc.device, weight_dtype)
        if acc.is_main_process:
            print(f"text prompts: {names} (null row {null_row}), "
                  f"dropout {args.text_dropout_prob}", flush=True)

    def text_for(names_or_rows, bsz, dropout=0.0):
        """(prompt_embeds, pooled) for a batch; falls back to the single fixed embedding."""
        if text_bank is None or names_or_rows is None:
            return null_prompt_embeds.repeat(bsz, 1, 1), null_pooled.repeat(bsz, 1)
        rows = names_or_rows
        if not torch.is_tensor(rows):
            rows = torch.tensor([text_row.get(d, null_row) for d in rows], device=acc.device)
        if dropout > 0:
            drop = torch.rand(rows.shape, device=rows.device) < dropout
            rows = torch.where(drop, torch.full_like(rows, null_row), rows)
        return text_bank[rows], text_pooled_bank[rows]

    avail_embedder = AvailEmbedder(null_pooled.shape[-1]) if args.use_avail else None
    if args.resume_from_checkpoint and avail_embedder is not None:
        ap = os.path.join(args.resume_from_checkpoint, "avail_embedder.pt")
        if os.path.exists(ap):
            avail_embedder.load_state_dict(torch.load(ap, map_location="cpu"))
            if acc.is_main_process:
                print(f"resumed avail_embedder from {ap}", flush=True)

    if vae is not None:
        vae.to(acc.device, dtype=weight_dtype)
    transformer.to(acc.device)
    if avail_embedder is not None:
        avail_embedder.to(acc.device)

    # The expanded I/O layers are the only parts that are not genuinely pretrained
    # (they were replicated from 16ch weights), so they need a much larger LR than the
    # 24 pretrained MMDiT blocks, which a high LR would just damage.
    new_layer_names = ("pos_embed.proj", "proj_out")
    # The conditioning path -- pooled projection -> time_text_embed -> every block's
    # AdaLN modulation -- sees an almost constant input: target_avail is [1,1,0] or
    # [1,1,1], and there are only a handful of prompts. A constant input means the
    # gradient points the same way on every step instead of averaging out over the
    # batch, so it accumulates without bound while the image path stays well behaved.
    # Attributing the gradient at a spike showed exactly that: time_text_embed owned
    # 82-85% of it and the AdaLN projections another ~20%, while the image path
    # (pos_embed.proj, attention, feed-forward) barely registered. Give that path its
    # own much smaller LR rather than letting it ride the backbone rate.
    cond_names = ("time_text_embed",)
    cond_params, new_params, base_params = [], [], []
    for n, p in transformer.named_parameters():
        if any(k in n for k in cond_names):
            cond_params.append(p)
        elif any(n.startswith(k) or f".{k}" in n for k in new_layer_names):
            new_params.append(p)
        else:
            base_params.append(p)
    if avail_embedder is not None:
        # feeds straight into time_text_embed, so it belongs to the same path
        cond_params += list(avail_embedder.parameters())
    params = base_params + new_params + cond_params
    cond_lr = args.learning_rate * args.cond_lr_mult
    groups = [
        {"params": base_params, "lr": args.learning_rate},
        {"params": new_params, "lr": args.learning_rate * args.new_layer_lr_mult},
        {"params": cond_params, "lr": cond_lr},
    ]
    print(f"param groups: backbone={len(base_params)} @ lr={args.learning_rate:g} | "
          f"new I/O={len(new_params)} @ lr={args.learning_rate * args.new_layer_lr_mult:g} | "
          f"conditioning={len(cond_params)} @ lr={cond_lr:g}")
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(groups, betas=(0.9, args.adam_beta2),
                                      weight_decay=1e-2, eps=1e-8)

    dataset = MRISliceDataset(args.latent_root if use_latents else args.data_root,
                              latent_mode=use_latents, datasets=args.datasets,
                              axial_only=args.axial_only,
                              out_of_plane_weight=args.out_of_plane_weight)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                        num_workers=args.dataloader_num_workers, drop_last=True, pin_memory=True)

    resume_step = 0
    if args.resume_from_checkpoint:
        resume_step = int(os.path.basename(
            os.path.normpath(args.resume_from_checkpoint)).split("-")[-1])
    sched_total = args.max_train_steps
    if args.resume_fresh_schedule and resume_step:
        sched_total = max(1, args.max_train_steps - resume_step)
    lr_sched = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * acc.num_processes,
        num_training_steps=sched_total * acc.num_processes,
    )

    resume_state_path = None
    if args.resume_from_checkpoint:
        cand = os.path.join(args.resume_from_checkpoint, "training_state.pt")
        if os.path.exists(cand):
            # full resume: optimizer + LR scheduler are loaded after acc.prepare().
            resume_state_path = cand
        else:
            # weights-only resume: the Adam moments restart at zero. Fast-forwarding the
            # schedule then drops full LR onto those cold moments, and both the v2 and v3
            # resumes diverged ~850 steps later -- the horizon over which a beta2=0.999
            # second moment is still an underestimate. --resume_fresh_schedule instead
            # re-runs warmup over the remaining steps, so the LR grows as the moments do.
            if args.resume_fresh_schedule:
                if acc.is_main_process:
                    print(f"resuming from step {resume_step} (weights only); FRESH schedule: "
                          f"{args.lr_warmup_steps} warmup + cosine over the remaining "
                          f"{args.max_train_steps - resume_step} steps", flush=True)
            else:
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
    latent_ch = acc.unwrap_model(transformer).config.in_channels
    latent_hw = args.resolution // 8          # SD3 VAE downsamples 8x

    @torch.no_grad()
    def run_validation(step):
        net = acc.unwrap_model(transformer)
        was_training = net.training
        net.eval()
        n = args.num_validation_samples
        sched_kwargs = {} if args.inference_shift is None else {"shift": args.inference_shift}
        # One block of samples per prompt. Every block starts from the SAME noise, so a
        # difference between blocks is the text doing something and nothing else.
        prompts = sorted(k for k in (text_row or {}) if k != "null") or [None]

        def sample_block(name):
            sched = FlowMatchEulerDiscreteScheduler.from_config(noise_scheduler.config, **sched_kwargs)
            sched.set_timesteps(args.num_inference_steps, device=acc.device)
            g = torch.Generator(device=acc.device).manual_seed(args.seed)
            lat = torch.randn((n, latent_ch, latent_hw, latent_hw), device=acc.device,
                              dtype=weight_dtype, generator=g)
            pe, pooled = text_for([name] * n if name else None, n)
            if avail_embedder is not None:
                # ask for a full [1,1,1] sample, i.e. a real (not copy-filled) FLAIR
                av = torch.ones((n, 3), device=acc.device, dtype=weight_dtype)
                pooled = pooled + acc.unwrap_model(avail_embedder)(av)
            for t in sched.timesteps:
                v = net(hidden_states=lat, timestep=t.expand(n).to(acc.device),
                        encoder_hidden_states=pe, pooled_projections=pooled, return_dict=False)[0]
                lat = sched.step(v, t, lat, return_dict=False)[0]
            lat = lat.to(weight_dtype) / val_vae.config.scaling_factor
            if args.n_modalities > 1:
                # decode each modality's own 16ch block separately, then stack as [n,3,H,W]
                ch = lat.shape[1] // args.n_modalities
                chans = []
                for k in range(args.n_modalities):
                    d = val_vae.decode(lat[:, k * ch:(k + 1) * ch]).sample   # [n,3,H,W] grayscale
                    chans.append(d.mean(1))                                  # -> [n,H,W]
                out = torch.stack(chans, dim=1)                              # [n,3,H,W]
            else:
                out = val_vae.decode(lat).sample
            return ((out + 1) / 2).clamp(0, 1).float().cpu().numpy()

        img = np.concatenate([sample_block(p) for p in prompts], axis=0)  # [len(prompts)*n,3,H,W]
        grid = np.concatenate(
            [np.concatenate([img[i, c] for c in range(3)], axis=1) for i in range(img.shape[0])],
            axis=0)
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
        fp = os.path.join(args.output_dir, "samples", f"step_{step:06d}.png")
        Image.fromarray((grid * 255).astype("uint8")).save(fp)
        if args.report_to == "wandb":
            import wandb
            rows = " -> ".join(f"{p or 'null'} x{n}" for p in prompts)
            acc.log({"samples": wandb.Image(
                fp, caption=f"step {step} — rows={rows}, cols=T1|T2|FLAIR")}, step=step)
        print(f"validation samples -> {fp}", flush=True)
        if was_training:
            net.train()

    if args.report_to == "wandb" and acc.is_main_process:
        acc.init_trackers(
            args.tracker_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    if acc.is_main_process:
        from collections import Counter
        by_ori = Counter(it.get("orientation", "axial") for it in dataset.items)
        print(f"dataset slices={len(dataset)} {dict(sorted(by_ori.items()))}  "
              f"subjects={len({(it['dataset'], it['subject']) for it in dataset.items})}  "
              f"steps={args.max_train_steps}  use_avail={args.use_avail}")
        if args.axial_only:
            print(f"axial-only sites: {list(args.axial_only)}")
        # effective supervision per modality x plane, after avail and the resolution weight
        eff = Counter()
        for it in dataset.items:
            o = it.get("orientation", "axial")
            rw = resolution_weight(it["dataset"], o, args.out_of_plane_weight)
            for mi, mod in enumerate(MODALITIES):
                eff[(mod, o)] += it["target_avail"][mi] * rw[mi]
        print(f"effective supervised slices (avail x resolution weight "
              f"{args.out_of_plane_weight:g}):")
        for mod in MODALITIES:
            print("   " + f"{mod:6}" + "".join(
                f"{o} {eff[(mod, o)]:>9,.0f}   " for o in ["axial", "coronal", "sagittal"]))

    global_step = resume_step
    loss_ema, n_skipped = None, 0
    gn_ema, n_grad_skipped = None, 0
    done = False
    while not done:
        for batch in loader:
            with acc.accumulate(transformer):
                if use_latents:
                    latents = batch["latents"].to(acc.device, dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(acc.device, dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
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

                pe, pooled = text_for(batch["orientation"], bsz, args.text_dropout_prob)
                if avail_embedder is not None:
                    pooled = pooled + avail_embedder(batch["avail"].to(acc.device, dtype=weight_dtype))

                model_pred = transformer(
                    hidden_states=noisy, timestep=timesteps,
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
                    ch_per_mod = target.shape[1] // args.n_modalities
                    m = batch["avail"].to(acc.device).float()
                    # down-weight (not drop) modalities being viewed across the slabs of
                    # a 2D acquisition. m stays a float weight and the normaliser is m.sum(),
                    # so this is still a correct weighted mean.
                    m = m * batch["res_weight"].to(acc.device).float()
                    m = m.repeat_interleave(ch_per_mod, dim=1)[:, :, None, None]
                    loss = (sq * m).sum() / (m.sum() * target.shape[-2] * target.shape[-1] + 1e-8)
                else:
                    loss = sq.reshape(bsz, -1).mean(1).mean()

                # Spike guard. The v1 run was healthy to step 20k (loss 0.19, clean
                # samples), then a burst of 10-100x losses drove it into a collapse it
                # never came back from -- gradient clipping does not save you here,
                # because Adam normalises the update anyway, so a run of bad directions
                # still corrupts the moments. Zeroing the loss keeps the batch from
                # reaching the optimizer at all, and the EMA only tracks accepted steps
                # so a spike storm cannot drag the threshold up behind it.
                lv = loss.detach().item()
                spike = (loss_ema is not None and args.loss_spike_factor > 0
                         and lv > args.loss_spike_factor * loss_ema)
                if spike:
                    n_skipped += 1
                    loss = loss * 0.0
                else:
                    loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv

                acc.backward(loss)
                grad_norm, do_step = None, True
                if acc.sync_gradients:
                    grad_norm = acc.clip_grad_norm_(params, args.max_grad_norm)
                    gn = float(grad_norm)
                    # The gradient is the instrument that matters: in the v2 run it
                    # crossed 10 at step 20880 while the loss was still a healthy 0.285,
                    # and the loss did not react for another 180 steps -- by which point
                    # the moments were already wrecked. Clipping alone does not save the
                    # run, because it rescales a direction that is itself garbage, so
                    # refuse the step outright.
                    spike_gn = (args.grad_spike_abs > 0 and gn > args.grad_spike_abs) or (
                        args.grad_spike_norm > 0 and gn_ema is not None
                        and gn > args.grad_spike_norm * gn_ema and gn > 2.0)
                    if spike_gn:
                        do_step, n_grad_skipped = False, n_grad_skipped + 1
                        # read the gradients BEFORE zeroing them: zero_grad(set_to_none=True)
                        # is the default, so afterwards there is nothing left to attribute
                        if acc.is_main_process and n_grad_skipped <= 200:
                            # WHERE is the gradient coming from? clip_grad_norm_ has already
                            # rescaled everything by the same factor, so the ranking is intact
                            # even though the magnitudes are not. If one layer owns the norm --
                            # especially one of the expanded I/O layers, which are the only
                            # parts that were never pretrained -- that is the bug, not the data.
                            net = acc.unwrap_model(transformer)
                            per = [(float(p.grad.norm()), n) for n, p in net.named_parameters()
                                   if p.grad is not None]
                            per.sort(reverse=True)
                            tot = sum(x * x for x, _ in per) ** 0.5 + 1e-12
                            top = "  ".join(f"{n}={100 * x / tot:.0f}%" for x, n in per[:4])
                            print(f"  [grad spike] step {global_step + 1} norm {gn:.1f} "
                                  f"(ema {gn_ema:.2f}) loss {lv:.3f} | top: {top}", flush=True)
                        optimizer.zero_grad()
                    else:
                        gn_ema = gn if gn_ema is None else 0.99 * gn_ema + 0.01 * gn
                if do_step:
                    optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if acc.sync_gradients:
                global_step += 1
                if args.report_to == "wandb":
                    logs = {"train_loss": lv, "lr": lr_sched.get_last_lr()[0],
                            "loss_ema": loss_ema or lv, "skipped_batches": n_skipped,
                            "grad_skipped": n_grad_skipped, "grad_ema": gn_ema or 0.0}
                    if grad_norm is not None:
                        logs["grad_norm"] = float(grad_norm)
                    acc.log(logs, step=global_step)
                if acc.is_main_process and global_step % args.log_every == 0:
                    gn = f"  grad {float(grad_norm):.2f}" if grad_norm is not None else ""
                    sk = (f"  skipped {n_skipped}/{n_grad_skipped}"
                          if (n_skipped or n_grad_skipped) else "")
                    print(f"step {global_step}/{args.max_train_steps}  loss {lv:.4f}{gn}{sk}",
                          flush=True)
                if (args.validation_steps and acc.is_main_process
                        and global_step % args.validation_steps == 0):
                    run_validation(global_step)
                if global_step % args.checkpointing_steps == 0 and acc.is_main_process:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    acc.unwrap_model(transformer).save_pretrained(os.path.join(save_dir, "transformer"))
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
