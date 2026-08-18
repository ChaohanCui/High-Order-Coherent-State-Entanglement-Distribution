# Source Code Provenance

This folder contains copies of the underlying research scripts used to produce
the selected data.

- `src/` and `scripts/` preserve the original runnable project layout for the
  selected simulations.
- `core/` and `drivers/` are duplicate convenience groupings of the same key
  files by role.
- `qam_source_loss_hashing.py` is included because the reflection-centered
  interface-loss source code imports its coherent-environment overlap helper.

The public-facing scripts in `../scripts/` are intentionally smaller and more
annotated.  They read curated data from `../data/raw` and produce the
per-subfigure CSV tables in `../outputs/subfigure_data`.

The source files here preserve provenance, but they are not all lightweight
reviewer commands.  In particular:

- 32-QAM scans can take a long time.
- 16-QAM interface-loss transition scans can take a long time.
- The optimized POVM search is nonconvex and depends on multi-start numerical
  optimization.
