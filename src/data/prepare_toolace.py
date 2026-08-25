#!/usr/bin/env python3
"""Convert official Hugging Face Team-ACE/ToolACE into Llama 3.1 tool-call chats."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, DatasetDict, load_dataset

TOOL_ARRAY_RE = re.compile(r"\[\{.*\}\]\s*$", re.DOTALL)
FUNC_CALL_RE = re.compile(r"^\[.*\]\s*$", re.DOTALL)
NAME_ARGS_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)\((?P<args>.*)\)$", re.DOTALL)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_tools_and_system(system_text: str, fallback: str) -> tuple[str, list[dict[str, Any]]]:
    text = (system_text or "").strip()
    match = TOOL_ARRAY_RE.search(text)
    tools: list[dict[str, Any]] = []
    if match:
        raw = match.group(0)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                tools = [normalize_tool(item) for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            tools = []
        system_prompt = text[: match.start()].strip()
    else:
        system_prompt = text
    if not system_prompt:
        system_prompt = fallback.strip()
    return system_prompt, tools


def normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters") or tool.get("input_schema") or {}
    parameters = rewrite_schema(parameters)
    name = tool.get("name") or tool.get("tool_name") or "unknown"
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": parameters,
        },
    }


def rewrite_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "type" and value == "dict":
                out[key] = "object"
            else:
                out[key] = rewrite_schema(value)
        return out
    if isinstance(node, list):
        return [rewrite_schema(item) for item in node]
    return node


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = depth_brack = depth_brace = 0
    quote: str | None = None
    escape = False
    for char in text:
        if quote:
            buf.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            buf.append(char)
            continue
        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack -= 1
        elif char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        if char == sep and depth_paren == depth_brack == depth_brace == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
            continue
        buf.append(char)
    piece = "".join(buf).strip()
    if piece:
        parts.append(piece)
    return parts


def parse_call_list(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    if not FUNC_CALL_RE.match(stripped):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return []
    calls = []
    for chunk in split_top_level(inner):
        parsed = parse_one_call(chunk)
        if parsed is None:
            return None
        calls.append(parsed)
    return calls


def parse_one_call(chunk: str) -> dict[str, Any] | None:
    match = NAME_ARGS_RE.match(chunk.strip())
    if not match:
        return None
    name = match.group("name")
    raw_args = match.group("args").strip()
    arguments: dict[str, Any] = {}
    if raw_args:
        for piece in split_top_level(raw_args):
            if "=" not in piece:
                return None
            key, value = piece.split("=", 1)
            key = key.strip()
            try:
                arguments[key] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                arguments[key] = value.strip().strip("'\"")
    return {
        "id": f"call_{abs(hash(name + json.dumps(arguments, default=str))) % 10_000_000}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def convert_example(example: dict[str, Any], fallback: str) -> dict[str, Any] | None:
    system_prompt, tools = extract_tools_and_system(example.get("system") or "", fallback)
    conversations = example.get("conversations") or []
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    pending_tool_ids: list[str] = []
    saw_assistant = False

    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        role = (turn.get("from") or turn.get("role") or "").lower()
        value = turn.get("value") if "value" in turn else turn.get("content")
        if value is None:
            value = ""
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)

        if role in {"human", "user"}:
            messages.append({"role": "user", "content": value})
        elif role in {"gpt", "assistant"}:
            calls = parse_call_list(value)
            if calls:
                pending_tool_ids = [call["id"] for call in calls]
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            else:
                pending_tool_ids = []
                messages.append({"role": "assistant", "content": value})
            saw_assistant = True
        elif role in {"tool", "function"}:
            tool_call_id = pending_tool_ids.pop(0) if pending_tool_ids else f"call_{len(messages)}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": value,
                }
            )
        else:
            return None

    if not saw_assistant:
        return None
    return {"messages": messages, "tools": tools}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fallback = cfg.get("system_fallback") or ""

    print(f"Loading {cfg['dataset_id']} from Hugging Face...", flush=True)
    raw = load_dataset(cfg["dataset_id"], split=cfg.get("split", "train"))
    converted: list[dict[str, Any]] = []
    dropped = 0
    for example in raw:
        item = convert_example(example, fallback)
        if item is None:
            dropped += 1
            continue
        converted.append(item)

    if not converted:
        print("No rows converted.", file=sys.stderr)
        return 1

    rng = random.Random(int(cfg.get("seed", 42)))
    rng.shuffle(converted)
    val_ratio = float(cfg.get("val_ratio", 0.05))
    n_val = max(1, int(len(converted) * val_ratio))
    val_rows = converted[:n_val]
    train_rows = converted[n_val:]

    dset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
        }
    )
    dset.save_to_disk(str(output_dir / "toolace_llama31"))
    stats = {
        "source": cfg["dataset_id"],
        "raw_rows": len(raw),
        "converted": len(converted),
        "dropped": dropped,
        "train": len(train_rows),
        "validation": len(val_rows),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    preview = converted[0]
    (output_dir / "preview.json").write_text(
        json.dumps({"messages": preview["messages"][:6], "n_tools": len(preview["tools"])}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    print(f"Wrote {output_dir / 'toolace_llama31'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
