"""Precompute the SD3 text embeddings for the conditioning prompts (Stage 0 + Stage 1).

Both stages used to run with ONE fixed embedding for every sample, so the text path
carried no information and text CFG was undefined -- both guidance branches saw the
same tokens and it cancelled. To make text guidance real, each prompt class gets its
own embedding and the empty string "" becomes the text-unconditional branch that
training drops to (see --text_dropout_prob in the trainers).

Writes a single dict:
  {name: {"prompt_embeds": [1,L,4096], "pooled_prompt_embeds": [1,2048]}, ...}
with "null" always present (the "" prompt). Run once on a GPU; output is a few MB.
"""
import argparse
import json

import torch
import torch.nn.functional as F
from transformers import (
    CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast, T5EncoderModel,
)

# One prompt per slice orientation. The dataset name is deliberately NOT in the
# prompt: it would let the model shortcut to a site identity instead of learning
# what the text actually has to control, and it is useless for a new site at
# inference. Orientation is the one attribute that genuinely changes the image and
# that you always know when you ask for a sample. "" is the text-unconditional
# branch that training drops to (--text_dropout_prob) and that CFG guides away from.
ORIENTATION_PROMPTS = {
    "null": "",
    "axial": "an axial slice of a 3T brain MRI",
    "coronal": "a coronal slice of a 3T brain MRI",
    "sagittal": "a sagittal slice of a 3T brain MRI",
}

# Legacy per-source set, kept so the older Stage 1 runs (which index the bank by
# dataset name) can still be reproduced.
SOURCE_PROMPTS = {
    "null": "",
    "kcl": "kcl axial ultra-low-field 64mT brain scan to 3T",
    "ulfenc": "ulfenc axial ultra-low-field 64mT brain scan to 3T",
    "webb": "webb axial ultra-low-field 64mT brain scan to 3T",
}

SCHEMES = {"orientation": ORIENTATION_PROMPTS, "source": SOURCE_PROMPTS}


def _clip(te, tok, prompt, dtype, device):
    ids = tok(prompt, padding="max_length", max_length=77, truncation=True,
              return_tensors="pt").input_ids
    out = te(ids.to(device), output_hidden_states=True)
    return out.hidden_states[-2].to(dtype=dtype, device=device), out[0]


def _t5(te, tok, prompt, max_len, dtype, device):
    ids = tok(prompt, padding="max_length", max_length=max_len, truncation=True,
              add_special_tokens=True, return_tensors="pt").input_ids
    return te(ids.to(device))[0].to(dtype=dtype, device=device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", default="orientation", choices=sorted(SCHEMES),
                    help='"orientation" (Stage 0 / current Stage 1: one prompt per slice '
                         'plane) or "source" (legacy: one prompt per dataset)')
    ap.add_argument("--prompts", default=None,
                    help='JSON dict {name: prompt}; overrides --scheme. '
                         '"null" is added automatically if missing.')
    ap.add_argument("--max_sequence_length", type=int, default=256)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    prompts = json.loads(args.prompts) if args.prompts else dict(SCHEMES[args.scheme])
    prompts.setdefault("null", "")

    device = "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    mp = args.pretrained_model_name_or_path

    tok1 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer_2")
    tok3 = T5TokenizerFast.from_pretrained(mp, subfolder="tokenizer_3")
    te1 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder").to(device, dtype)
    te2 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder_2").to(device, dtype)
    te3 = T5EncoderModel.from_pretrained(mp, subfolder="text_encoder_3").to(device, dtype)

    out = {}
    for name, prompt in prompts.items():
        e1, p1 = _clip(te1, tok1, prompt, dtype, device)
        e2, p2 = _clip(te2, tok2, prompt, dtype, device)
        clip_embeds = torch.cat([e1, e2], dim=-1)
        pooled = torch.cat([p1, p2], dim=-1)
        t5 = _t5(te3, tok3, prompt, args.max_sequence_length, dtype, device)
        clip_embeds = F.pad(clip_embeds, (0, t5.shape[-1] - clip_embeds.shape[-1]))
        out[name] = {"prompt_embeds": torch.cat([clip_embeds, t5], dim=-2).cpu(),
                     "pooled_prompt_embeds": pooled.cpu(),
                     "prompt": prompt}
        print(f"{name:8} '{prompt}' -> {tuple(out[name]['prompt_embeds'].shape)}")

    torch.save(out, args.out)
    print(f"\nsaved {len(out)} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
