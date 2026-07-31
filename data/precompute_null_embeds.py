"""
Precompute the fixed NULL ("") SD3 text embeddings once and save to disk, so the
unconditional trainer never loads the text encoders (T5-XXL alone is ~9.5GB).

Loads the 3 SD3 text encoders + tokenizers, encodes "", saves
{prompt_embeds, pooled_prompt_embeds} to <out>. Run once in a GPU container
(needs SD3 access). Tiny output (~a few MB).
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import (
    CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast, T5EncoderModel,
)


def _clip(te, tok, prompt, dtype, device):
    ids = tok(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    out = te(ids.to(device), output_hidden_states=True)
    return out.hidden_states[-2].to(dtype=dtype, device=device), out[0]


def _t5(te, tok, prompt, max_len, dtype, device):
    ids = tok(prompt, padding="max_length", max_length=max_len, truncation=True,
              add_special_tokens=True, return_tensors="pt").input_ids
    return te(ids.to(device))[0].to(dtype=dtype, device=device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--out", default="/home/rintern14/ymk/data_stage0_3t_latents/null_embeds.pt")
    ap.add_argument("--prompt", default="", help='fixed prompt to embed (Stage 1 uses "enhance")')
    ap.add_argument("--max_sequence_length", type=int, default=256)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    device = "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    mp = args.pretrained_model_name_or_path

    tok1 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(mp, subfolder="tokenizer_2")
    tok3 = T5TokenizerFast.from_pretrained(mp, subfolder="tokenizer_3")
    te1 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder").to(device, dtype)
    te2 = CLIPTextModelWithProjection.from_pretrained(mp, subfolder="text_encoder_2").to(device, dtype)
    te3 = T5EncoderModel.from_pretrained(mp, subfolder="text_encoder_3").to(device, dtype)

    e1, p1 = _clip(te1, tok1, args.prompt, dtype, device)
    e2, p2 = _clip(te2, tok2, args.prompt, dtype, device)
    clip_embeds = torch.cat([e1, e2], dim=-1)
    pooled = torch.cat([p1, p2], dim=-1)
    t5 = _t5(te3, tok3, args.prompt, args.max_sequence_length, dtype, device)
    clip_embeds = F.pad(clip_embeds, (0, t5.shape[-1] - clip_embeds.shape[-1]))
    prompt_embeds = torch.cat([clip_embeds, t5], dim=-2)

    torch.save({"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled.cpu()}, args.out)
    print(f"saved {args.out}  prompt_embeds={tuple(prompt_embeds.shape)} pooled={tuple(pooled.shape)}")


if __name__ == "__main__":
    main()
