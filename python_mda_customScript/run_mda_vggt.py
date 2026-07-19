# -*- coding: utf-8 -*-
"""MDA VGGT reconstruction entry point with VGGT-compatible JSON/GLB output."""

from __future__ import annotations

import os
import sys

MODEL_NAME = "vggt_mog_l2"
DEFAULT_OUTPUT_TAG = "mda_vggt"

os.environ.setdefault("MDA_DEFAULT_MODEL_NAME", MODEL_NAME)
os.environ.setdefault("MDA_DEFAULT_OUTPUT_TAG", DEFAULT_OUTPUT_TAG)

from run_mda_da3 import main  # noqa: E402


def option_value(option: str) -> str | None:
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == option:
            return sys.argv[index + 1] if index + 1 < len(sys.argv) else None
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


if __name__ == "__main__":
    requested_model = option_value("--model-name")
    if requested_model is not None and requested_model != MODEL_NAME:
        raise SystemExit(
            f"run_mda_vggt.py only supports --model-name {MODEL_NAME}, "
            f"not {requested_model}."
        )
    if requested_model is None:
        sys.argv.extend(["--model-name", MODEL_NAME])
    if option_value("--output-tag") is None:
        sys.argv.extend(["--output-tag", DEFAULT_OUTPUT_TAG])
    main()
