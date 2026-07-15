#!/usr/bin/env python
"""MDA workspace entry point for the non-commercial VGGT checkpoint."""

from run_vggt_common import main


if __name__ == "__main__":
    main(
        model_id="facebook/VGGT-1B",
        model_display_name="VGGT-1B",
        commercial=False,
    )
