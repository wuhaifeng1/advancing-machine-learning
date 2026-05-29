#!/usr/bin/env python3
"""Run the lightweight cross-robot loop matching analysis.

This entry point intentionally avoids ROS and supervised training. It uses only
the compact GICP-passed evidence exported from Co-LRIO and simulates the report
pipeline:

  Scan Context evidence -> GICP-passed evidence -> PCM/final-used filtering
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.loop_matching_analysis import main


if __name__ == "__main__":
    main()
