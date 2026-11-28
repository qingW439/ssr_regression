"""Single source of truth for asset paths (code lives in the repo, assets outside)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
DATA = ASSETS / 'data/emobank_processed'
UVR = ASSETS / 'pretrained/roberta-base-simcse'
TARGETS = ASSETS / 'm1_v2_assets/targets'
FOLD_MANIFESTS = ASSETS / 'Semi-Supervised-Regression-avrrc/outputs'
TEACHERS = ASSETS / 'm1_hdt_gate/teachers'
FUSED = ASSETS / 'm1_hdt_gate/fused'
RUNS = ASSETS / 'm1_runs_unified'
