#!/usr/bin/env python3
"""Fine-tune on ToolACE with BFCL Llama-3.1-FC prompt/target alignment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")
os.environ.setdefault("CUDA_HOME", _cuda_home)
os.environ["PATH"] = f"{_cuda_home}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{_cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_messages_tools(example: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "messages_json" in example:
        messages = json.loads(example["messages_json"])
        tools = json.loads(example.get("tools_json") or "[]")
    else:
        messages = list(example["messages"])
        tools = example.get("tools") or []
    return messages, tools


def render_bfcl_llama31(example: dict[str, Any]) -> str:
    """Match bfcl_eval LlamaHandler_3_1._format_prompt + gold assistant turns."""
    messages, tools = _load_messages_tools(example)

    system_message = ""
    remaining = messages
    if messages and messages[0].get("role") == "system":
        system_message = (messages[0].get("content") or "").strip()
        remaining = messages[1:]

    parts: list[str] = ["<|begin_of_text|>"]
    parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
    if tools:
        parts.append("Environment: ipython\n")
    parts.append("Cutting Knowledge Date: December 2023\n")
    parts.append("Today Date: 26 Jul 2024\n\n")
    parts.append(system_message)
    parts.append("<|eot_id|>")

    first_user = True
    for message in remaining:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "user" and first_user:
            first_user = False
            parts.append("<|start_header_id|>user<|end_header_id|>\n\n")
            if tools:
                parts.append(
                    "Given the following functions, please respond with a JSON for a function call "
                    "with its proper arguments that best answers the given prompt.\n\n"
                )
                parts.append(
                    'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
                )
                parts.append("Do not use variables.\n\n")
                for func in tools:
                    parts.append(json.dumps(func, indent=4))
                    parts.append("\n\n")
            parts.append(f"{content}<|eot_id|>")
        elif role == "tool":
            parts.append("<|start_header_id|>ipython<|end_header_id|>\n\n")
            parts.append(content)
            parts.append("<|eot_id|>")
        elif role in {"assistant", "user", "system", "ipython"}:
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        else:
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

    text = "".join(parts)
    # Drop empty assistant targets (should be filtered in prepare; skip quietly in map).
    if "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>" in text:
        return ""
    return text


def render_example(example: dict[str, Any], tokenizer: AutoTokenizer | None = None) -> dict[str, str]:
    del tokenizer  # BFCL layout is hand-built; tokenizer chat_template is unreliable on ToolACE-8B.
    return {"text": render_bfcl_llama31(example)}


def build_model(cfg: dict[str, Any]):
    method = cfg["method"]
    model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if method == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    load_kwargs = {
        "quantization_config": quant_config,
        "device_map": "auto",
        "attn_implementation": cfg.get("attn_implementation", "eager"),
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            **load_kwargs,
        )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = None
    if method in {"lora", "qlora"}:
        lora_cfg = cfg["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg["target_modules"]),
        )
    return model, tokenizer, peft_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_root = Path(cfg["data_dir"]) / "toolace_llama31"
    dset = load_from_disk(str(data_root))
    model, tokenizer, peft_config = build_model(cfg)

    # Spot-check: at least some rendered rows must contain BFCL JSON tool calls.
    json_hits = 0
    for i in range(min(50, len(dset["train"]))):
        sample = render_example(dset["train"][i])["text"]
        if '"name"' in sample and '"parameters"' in sample:
            json_hits += 1
    if json_hits < 5:
        raise SystemExit(f"Train render sanity failed: only {json_hits}/50 rows have BFCL JSON targets")

    train = dset["train"].map(
        lambda row: render_example(row),
        remove_columns=dset["train"].column_names,
        desc="render-train-bfcl",
    )
    val = dset["validation"].map(
        lambda row: render_example(row),
        remove_columns=dset["validation"].column_names,
        desc="render-val-bfcl",
    )
    train = train.filter(lambda row: bool(row["text"]))
    val = val.filter(lambda row: bool(row["text"]))

    # Extra guard: reject empty assistant blocks at dataset scale.
    empty = sum(
        1
        for t in train["text"][:200]
        if "<|start_header_id|>assistant<|end_header_id|>\n\n<|eot_id|>" in t
    )
    if empty:
        raise SystemExit(f"Found {empty}/200 empty assistant targets — aborting train")
    print(f"Rendered train={len(train)} val={len(val)} json_hits_in_50={json_hits}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    batch = int(cfg["per_device_train_batch_size"]) * int(cfg["gradient_accumulation_steps"])
    steps_per_epoch = max(1, len(train) // batch)
    total_steps = steps_per_epoch * int(cfg["num_train_epochs"])
    warmup_steps = max(1, int(total_steps * float(cfg.get("warmup_ratio", 0.03))))

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=cfg.get("run_name"),
        seed=int(cfg.get("seed", 42)),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        bf16=bool(cfg.get("bf16", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        max_length=int(cfg.get("max_seq_length", 8192)),
        packing=bool(cfg.get("packing", False)),
        dataset_text_field="text",
        assistant_only_loss=False,
        report_to=[],
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
