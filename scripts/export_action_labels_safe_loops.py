"""Compatibility entry; validated loop/recovery fixes now use the main backend.

Accepts ordinary tools/export_action_labels.py arguments without selecting an older
isolated runtime. Prefer the main export entry for new processing.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import export_action_labels as exporter
if __name__=='__main__':
    print('Loop/recovery fixes merged: using the current local native ORB-SLAM3 backend.', flush=True)
    raise SystemExit(exporter.main())
