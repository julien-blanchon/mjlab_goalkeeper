#!/usr/bin/env -S uv --quiet run --script
# /// script
# dependencies = [
#   "mjlab",
#   "mjlab-goalkeeper",
# ]
# [tool.uv.sources]
# mjlab = { git = "https://github.com/julien-blanchon/mjlab" }
# mjlab-goalkeeper = { path = ".", editable = true }
# ///
"""Registers the custom goalkeeper task before running mjlab's training pipeline."""

import mjlab_goalkeeper  # noqa: F401

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
