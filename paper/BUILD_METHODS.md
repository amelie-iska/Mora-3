# Rebuilding the complete methods manuscript

Canonical prose and equations: `paper/mora3_complete_methods.md`.
Compatibility manuscript path: `paper/mora3_preprint.md` (generated exact copy).
Live offline-first paper: `docs/paper.html`.
Print paper: `paper/mora3_methods.pdf`, also copied to `docs/mora3_methods.pdf`.

The builder uses the existing project Python environment, its installed `markdown`
and Playwright packages, local `/opt/google/chrome/chrome`, `pdflatex`, `pdftocairo`,
`pdfinfo` and `pdftotext`. It installs nothing and uses no external math/diagram CDN.
The six architecture/mathematical SVGs and typeset equation images are
built from the reviewed prose and generator. The exact equation count is recorded
in `paper/methods_build_manifest.json`; do not infer it from this description.

From a reviewed isolated checkout, using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES= \
  /home/iska/miniconda3/envs/tokengt/bin/python -B \
  tools/build_complete_methods_paper.py \
  --output outputs/methods-paper-reviewed --pdf

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /home/iska/miniconda3/envs/tokengt/bin/python -B \
  tests/project/test_complete_methods_paper.py
```

The native proxy grid supplies only preregistered row labels and hyperparameter
values. Every result field is `null`; every empirical chart series is empty.
No run history is loaded, no model is trained, no optimizer is invoked and no
publication occurs. Original source/implementation evidence is not converted into
trained performance. Future result ingestion requires exact approved campaign,
checkpoint, split, metric and denominator references; it is intentionally absent
from this unpopulated manuscript build.

The live diagrams support keyboard selection, bounded construction-prefix replay,
a declared-square relation toggle and SVG export. Static figures remain complete
in the PDF, and old paper-section anchors have compatible destinations. The PDF
uses a tagged, outlined print when supported by the already installed browser.
Inspect all rendered pages before accepting a regenerated print artifact.

Do not upload the supplied Grandis reference book or private project instructions.
Publish only reviewed source checkpoints under the current explicit user authorization.
The manuscript and passing tests do not themselves confer training admission.


## Current publication directive — 14 November 2026

The user now authorizes pushing reviewed, tested source and manuscript changes at meaningful progress checkpoints without waiting for training. This supersedes the earlier publication condition above, but does not change source rights, the 24+2 run cap, training admission or the meaning of empty result cells. Historical receipts retain their original policy snapshot.
