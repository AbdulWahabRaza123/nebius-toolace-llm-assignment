#!/usr/bin/env python3
"""Merge a LoRA/QLoRA adapter into the base Llama 3.1 checkpoint for vLLM / Token Factory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))
    print(f"Merged checkpoint written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
