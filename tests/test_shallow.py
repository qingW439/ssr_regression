#!/usr/bin/env python3
"""CPU checks for the 6-layer shallow encoder (no training).

  S1  exactly 6 encoder layers exist and config.num_hidden_layers == 6.
  S2  no parameters from layers 6..11 survive in the state dict.
  S3  shallow last_hidden_state == full model hidden_states[6] (same UVR weights).
  S4  the regression head is structurally identical to U1 (same keys/shapes).
  S5  forward is finite and deterministic in eval mode.
"""
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO / 'src/hdtm'))
from train_shallow import ShallowVAD  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402

fail = []


def check(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail, flush=True)
    if not ok:
        fail.append(name)


def main():
    torch.set_num_threads(4)
    UVR = str(ROOT / 'pretrained/roberta-base-simcse')
    full = RobertaVADRegressor(UVR, dropout=0.1, offline=True).eval()
    sh = ShallowVAD(UVR, n_layers=6, dropout=0.1, offline=True).eval()

    check('S1_layer_count', len(sh.encoder.encoder.layer) == 6
          and sh.encoder.config.num_hidden_layers == 6)
    full_keys = set(full.state_dict())
    sh_keys = set(sh.state_dict())
    deep = [k for k in full_keys - sh_keys if 'encoder.layer.' in k]
    check('S2_deep_layers_removed', all(int(k.split('encoder.layer.')[1].split('.')[0]) >= 6 for k in deep)
          and len(deep) > 0, f'removed={len(deep)} keys')
    check('S2_no_new_keys', not (sh_keys - full_keys), f'extra={sorted(sh_keys - full_keys)[:3]}')
    head_ok = all(full.state_dict()[k].shape == sh.state_dict()[k].shape
                  for k in full_keys if k.startswith('head.'))
    check('S4_head_matches_u1', head_ok)

    ids = torch.randint(0, 30000, (3, 20))
    am = torch.ones_like(ids)
    am[1, 15:] = 0
    with torch.inference_mode():
        out_full = full.encoder(input_ids=ids, attention_mask=am, return_dict=True,
                                output_hidden_states=True).hidden_states
        out_sh = sh.encoder(input_ids=ids, attention_mask=am, return_dict=True).last_hidden_state
    diff = (out_full[6] - out_sh).abs().max().item()
    check('S3_prefix_matches_full', diff < 1e-6, f'maxdiff={diff:.3e}')

    with torch.inference_mode():
        a = sh(input_ids=ids, attention_mask=am)
        b = sh(input_ids=ids, attention_mask=am)
    check('S5_forward_finite', tuple(a.shape) == (3, 3) and bool(torch.isfinite(a).all()))
    check('S5_eval_deterministic', torch.equal(a, b))

    p_full = sum(p.numel() for p in full.parameters())
    p_sh = sum(p.numel() for p in sh.parameters())
    print(f'params full={p_full} shallow={p_sh} ratio={p_sh / p_full:.4f}', flush=True)
    check('S1_param_reduction', p_sh < 0.75 * p_full, f'{p_sh / p_full:.3f} of full')

    print('RESULT', 'ALL_PASS' if not fail else 'FAILED:' + ','.join(fail), flush=True)
    return 0 if not fail else 1


if __name__ == '__main__':
    raise SystemExit(main())
