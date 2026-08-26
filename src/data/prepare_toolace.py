#!/usr/bin/env python3
"""Convert Team-ACE/ToolACE into BFCL Llama-3.1-FC aligned training chats.

BFCL's LlamaHandler_3_1 expects assistant outputs like:
  {"name": "fn", "parameters": {...}}
or a list / semicolon-separated list of those objects.
ToolACE-8B's stripped chat template drops tool_calls, so we store the JSON
as plain assistant content and render prompts with the same BFCL layout.
"""

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

FUNC_CALL_RE = re.compile(r"^\[.*\]\s*$", re.DOTALL)
# ToolACE names often include spaces (e.g. "Market Trends API") or leading "/".
NAME_ARGS_RE = re.compile(r"^(?P<name>[A-Za-z0-9_./\s-]+?)\((?P<args>.*)\)$", re.DOTALL)
TOOLACE_EPILOGUE_RE = re.compile(
    r"Should you decide to return the function call\(s\).*$",
    re.DOTALL | re.IGNORECASE,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_json_array_span(text: str) -> tuple[int, int] | None:
    """Find the first top-level JSON array span (handles trailing ToolACE prose)."""
    start = text.find("[{")
    if start < 0:
        start = text.find("[")
        if start < 0:
            return None
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {'"', "'"}:
            in_str = True
            quote = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def extract_tools_and_system(system_text: str, fallback: str) -> tuple[str, list[dict[str, Any]]]:
    text = (system_text or "").strip()
    tools: list[dict[str, Any]] = []
    span = find_json_array_span(text)
    if span:
        start, end = span
        raw = text[start:end]
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                tools = [normalize_tool_bfcl(item) for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            tools = []
        head = text[:start].strip()
        tail = text[end:].strip()
        tail = TOOLACE_EPILOGUE_RE.sub("", tail).strip()
        system_prompt = "\n".join(p for p in (head, tail) if p).strip()
    else:
        system_prompt = TOOLACE_EPILOGUE_RE.sub("", text).strip()
    if not system_prompt:
        system_prompt = fallback.strip()
    # Keep system short; BFCL injects its own FC instructions in the user turn.
    if len(system_prompt) > 1200:
        system_prompt = system_prompt[:1200].rsplit(" ", 1)[0]
    return system_prompt, tools


def normalize_tool_bfcl(tool: dict[str, Any]) -> dict[str, Any]:
    """BFCL Llama FC expects flat {name, description, parameters}."""
    if "function" in tool and isinstance(tool["function"], dict):
        tool = tool["function"]
    parameters = tool.get("parameters") or tool.get("input_schema") or {}
    parameters = rewrite_schema(parameters)
    return {
        "name": tool.get("name") or tool.get("tool_name") or "unknown",
        "description": tool.get("description") or "",
        "parameters": parameters,
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
    return {"name": name, "parameters": arguments}


def dumps_for_bfcl_eval(obj: Any) -> str:
    """Serialize so BFCL's eval()-based decoder accepts bools/null."""

    def _fmt(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return repr(value) if isinstance(value, int) else json.dumps(value)
        if isinstance(value, list):
            return "[" + ", ".join(_fmt(v) for v in value) + "]"
        if isinstance(value, dict):
            items = []
            for k, v in value.items():
                items.append(f"{json.dumps(str(k), ensure_ascii=False)}: {_fmt(v)}")
            return "{" + ", ".join(items) + "}"
        return json.dumps(value, ensure_ascii=False)

    return _fmt(obj)


def calls_to_assistant_content(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return ""
    if len(calls) == 1:
        return dumps_for_bfcl_eval(calls[0])
    # BFCL accepts a JSON list or semicolon-separated objects.
    return dumps_for_bfcl_eval(calls)


def convert_example(example: dict[str, Any], fallback: str) -> dict[str, Any] | None:
    system_prompt, tools = extract_tools_and_system(example.get("system") or "", fallback)
    conversations = example.get("conversations") or []
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    saw_assistant = False
    saw_tool_call = False

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
            if calls is not None:
                if not calls:
                    # Explicit empty call list → refusal / no-tool answer.
                    content = "None of the functions can be used to answer this question."
                else:
                    content = calls_to_assistant_content(calls)
                    saw_tool_call = True
                messages.append({"role": "assistant", "content": content})
            else:
                content = value.strip()
                if not content:
                    return None
                messages.append({"role": "assistant", "content": content})
            saw_assistant = True
        elif role in {"tool", "function"}:
            messages.append({"role": "tool", "content": value})
        else:
            return None

    if not saw_assistant:
        return None
    # Drop rows where we expected tools but failed to extract any schema.
    if saw_tool_call and not tools:
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
    n_with_tools = 0
    n_with_json_assistant = 0
    for example in raw:
        item = convert_example(example, fallback)
        if item is None:
            dropped += 1
            continue
        converted.append(item)
        if item["tools"]:
            n_with_tools += 1
        for msg in item["messages"]:
            if msg["role"] == "assistant" and '"name"' in (msg.get("content") or ""):
                n_with_json_assistant += 1
                break

    if not converted:
        print("No rows converted.", file=sys.stderr)
        return 1

    rng = random.Random(int(cfg.get("seed", 42)))
    rng.shuffle(converted)
    val_ratio = float(cfg.get("val_ratio", 0.05))
    n_val = max(1, int(len(converted) * val_ratio))
    val_rows = converted[:n_val]
    train_rows = converted[n_val:]

    def flatten_row(row: dict[str, Any]) -> dict[str, str]:
        # Serialize nested fields: tool schemas are heterogeneous and break Arrow.
        return {
            "messages_json": json.dumps(row["messages"], ensure_ascii=False),
            "tools_json": json.dumps(row["tools"], ensure_ascii=False),
        }

    dset = DatasetDict(
        {
            "train": Dataset.from_list([flatten_row(r) for r in train_rows]),
            "validation": Dataset.from_list([flatten_row(r) for r in val_rows]),
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
        "rows_with_tools": n_with_tools,
        "rows_with_json_assistant": n_with_json_assistant,
        "format": "bfcl_llama31_fc_json",
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    preview = converted[0]
    (output_dir / "preview.json").write_text(
        json.dumps(
            {
                "messages": preview["messages"][:6],
                "n_tools": len(preview["tools"]),
                "tool0": preview["tools"][0] if preview["tools"] else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    print(f"Wrote {output_dir / 'toolace_llama31'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
