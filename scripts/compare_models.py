#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CompareConfig:
    base_model: str
    trained_model: str
    prompts_path: Optional[str]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    device: str
    dtype: str
    batch_size: int
    eos_token: Optional[str]


def load_model_and_tokenizer(model_path: str, device: str, dtype: str):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device)
    model.eval()
    return model, tok


@torch.no_grad()
def generate_text(model, tok, prompts: List[str], cfg: CompareConfig) -> List[str]:
    outputs: List[str] = []
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        gen_ids = model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=(cfg.temperature > 0),
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend(texts)
    return outputs


@torch.no_grad()
def compute_logprobs(model, tok, prompts: List[str], responses: List[str], cfg: CompareConfig) -> torch.Tensor:
    assert len(prompts) == len(responses)
    all_logps: List[torch.Tensor] = []
    for i in range(0, len(prompts), cfg.batch_size):
        p_batch = prompts[i : i + cfg.batch_size]
        r_batch = responses[i : i + cfg.batch_size]
        inputs = tok(p_batch, return_tensors="pt", padding=True, truncation=True)
        # Build labels as prompt+response; mask prompt tokens to -100
        resp_tok = tok(r_batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = torch.cat([inputs.input_ids, resp_tok.input_ids[:, : cfg.max_new_tokens]], dim=1)
        attention_mask = torch.cat([inputs.attention_mask, resp_tok.attention_mask[:, : cfg.max_new_tokens]], dim=1)

        # Align lengths with manual right padding
        max_len = int(attention_mask.sum(dim=1).max().item())
        pad_id = tok.pad_token_id
        if input_ids.size(1) < max_len:
            pad = torch.full((input_ids.size(0), max_len - input_ids.size(1)), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros((attention_mask.size(0), pad.size(1)), dtype=attention_mask.dtype)], dim=1
            )

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (bs, seqlen, vocab)

        # Compute token-level logprobs of responses
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)

        # labels: next tokens
        labels = input_ids[:, 1:]
        # gather logprobs at label positions
        token_logps = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # mask out prompt positions
        prompt_len = inputs.input_ids.size(1)
        resp_len = labels.size(1) - (prompt_len - 1)
        mask = torch.zeros_like(token_logps, dtype=torch.bool)
        mask[:, prompt_len - 1 : prompt_len - 1 + resp_len] = True
        token_logps = torch.where(mask, token_logps, torch.zeros_like(token_logps))
        all_logps.append(token_logps)

    return torch.cat(all_logps, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Compare base and trained models on generations and log-probs")
    ap.add_argument("--base_model", required=True, help="HF repo or local path of original model")
    ap.add_argument("--trained_model", required=True, help="HF repo or local path of trained model")
    ap.add_argument("--prompts_path", default=None, help="JSON list or JSONL file of prompts")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eos_token", default=None)
    ap.add_argument("--out", default="compare_output.jsonl")
    args = ap.parse_args()

    cfg = CompareConfig(
        base_model=args.base_model,
        trained_model=args.trained_model,
        prompts_path=args.prompts_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        eos_token=args.eos_token,
    )

    # Load prompts
    if cfg.prompts_path is None:
        prompts = [
            "Write a short poem about the sea.",
            "What is the capital of France?",
            "Explain the concept of reinforcement learning in one paragraph.",
            "Translate to Spanish: 'Good morning, how are you?'",
        ]
    else:
        if cfg.prompts_path.endswith(".jsonl"):
            prompts = [json.loads(line)["prompt"] for line in open(cfg.prompts_path)]
        else:
            items = json.load(open(cfg.prompts_path))
            prompts = [item["prompt"] if isinstance(item, dict) else item for item in items]

    base_model, base_tok = load_model_and_tokenizer(cfg.base_model, cfg.device, cfg.dtype)
    trained_model, trained_tok = load_model_and_tokenizer(cfg.trained_model, cfg.device, cfg.dtype)

    # Optional: enforce same eos_token
    if cfg.eos_token is not None:
        base_tok.eos_token = cfg.eos_token
        trained_tok.eos_token = cfg.eos_token

    # Generate
    base_outputs = generate_text(base_model, base_tok, prompts, cfg)
    trained_outputs = generate_text(trained_model, trained_tok, prompts, cfg)

    # Compute log-probs of generated responses under each model
    base_logps = compute_logprobs(base_model, base_tok, prompts, base_outputs, cfg)
    trained_logps = compute_logprobs(trained_model, trained_tok, prompts, trained_outputs, cfg)

    # Aggregate per-sample metrics
    with open(args.out, "w") as f:
        for i, prompt in enumerate(prompts):
            rec = {
                "prompt": prompt,
                "base_text": base_outputs[i],
                "trained_text": trained_outputs[i],
                "base_mean_logp": float(base_logps[i].sum().item() / (base_logps[i] != 0).sum().clamp(min=1).item()),
                "trained_mean_logp": float(trained_logps[i].sum().item() / (trained_logps[i] != 0).sum().clamp(min=1).item()),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()


