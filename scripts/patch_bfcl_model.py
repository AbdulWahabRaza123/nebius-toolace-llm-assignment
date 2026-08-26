#!/usr/bin/env python3
"""Patch BFCL model registry for our local vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path.home() / "gorilla" / "berkeley-function-call-leaderboard"
    cfg = root / "bfcl_eval" / "constants" / "model_config.py"
    text = cfg.read_text(encoding="utf-8")
    marker = '"llama-3.1-8b-toolace-FC"'
    if marker in text:
        print("already patched")
        return 0

    # Prefer reusing an existing OpenAI-compatible OSS handler already imported in this file.
    patch = '''
    "llama-3.1-8b-toolace-FC": ModelConfig(
        model_name="llama-3.1-8b-toolace",
        display_name="Llama-3.1-8B-ToolACE (FC) (local vLLM)",
        url="local",
        org="local",
        license="llama3.1",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
'''
    # Insert before the closing of MODEL_CONFIG_MAPPING if possible.
    if "MODEL_CONFIG_MAPPING = {" not in text:
        raise SystemExit("MODEL_CONFIG_MAPPING not found")

    # Append near the end of the dict by replacing the last occurrence of "}\n".
    idx = text.rfind("}")
    if idx < 0:
        raise SystemExit("could not find insertion point")
    # Find last mapping entry closing before final brace of the file's mapping.
    # Safer: insert after first entry of MODEL_CONFIG_MAPPING.
    anchor = "MODEL_CONFIG_MAPPING = {"
    pos = text.find(anchor) + len(anchor)
    text = text[:pos] + "\n" + patch + text[pos:]
    cfg.write_text(text, encoding="utf-8")
    print(f"patched {cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
