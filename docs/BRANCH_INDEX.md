# BRANCH_INDEX — historical branches and their fate

GitHub: `qingW439/ssr_regression` (public). Default branch `rankup-reproduction` is kept frozen.
All other branches were audited, their text assets extracted into `archive/extracted/`, and then
deleted after the bundle backup was verified.

Backup: `backups/ssr_pre_consolidation_20261128.bundle` (54 MB, 24 refs, complete history)
SHA256 `832991095267efa33088d30af6f841f37734c02474f30cd68383d99163eee923`

| branch | last commit | files | unique commits (vs all others) | content | extracted to | fate |
| --- | --- | --- | --- | --- | --- | --- |
| rankup-reproduction | b5b1747 | 186 | 2 | RankUp Yelp reproduction baseline | — | **kept (default, frozen)** |
| emobank-rankup-3d | af2b3f9 | 279 | 3–11 | EmoBank 3-D adaptation, label budgets | `archive/extracted/emobank-rankup-3d/` | deleted after extraction |
| emobank-ssr-rcus-3d | b6801cb | 297 | 3–14 | RCUS 3-D adaptation | `archive/extracted/emobank-ssr-rcus-3d/` | deleted after extraction |
| emobank-strong-supervised | be669b6 | 366 | 3–14 | strong supervised + fold manifests | `archive/extracted/emobank-strong-supervised/` | deleted after extraction |
| m1/avrrc-oof | 44c247a | 1123 | 16–61 | early M1, OOF pipeline, ledgers | `archive/extracted/m1_avrrc-oof/` | deleted after extraction |
| archive/wq-20261114 | caf58a4 | 26712 | 22–68 | JACF/RDA/UVR archive, reports | `archive/extracted/archive_wq-20261114/` | deleted after extraction |
| archive/legacy-20261104 | 324c341 | 208 | 38–84 | original default branch snapshot | `archive/extracted/archive_legacy-20261104/` | deleted after extraction |

New long-lived branch: **wq-main** (all future M1/M2/M3 work). Version tags:

- `v0.1.0-hdtm` — HDT-M formal 15-fold baseline (this commit).

Extraction rule: `repro_records/`, `results/`, `config/`, `docs/` files ≤ 2 MB, text formats only.
No weights, datasets, environments or caches were extracted.
