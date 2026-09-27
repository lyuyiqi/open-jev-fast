Scripts that generate the figures and the PDF report in `docs/` (matplotlib + reportlab).

- `charts.py` writes `ladder.{png,svg}` and `jevbench_cdf.{png,svg}` from `results/`. Run it from `docs/src/`.
- `make_public_en.py` builds `report.pdf`; it expects the figures as `ladder_en.png` and `cdf_en.png` next to it.

Both expect `NotoSansSC-Regular.ttf` and `NotoSansSC-Bold.ttf` (Noto Sans SC, SIL Open Font License, from Google Fonts) one directory above the working directory.
