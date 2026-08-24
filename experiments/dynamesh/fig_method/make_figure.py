#!/usr/bin/env python3
"""Build the MCFM + rung27 method figure.

    python3 make_figure.py                       -> fig_mcfm_rung27.svg
    inkscape -z -A fig_mcfm_rung27.pdf --export-text-to-path fig_mcfm_rung27.svg
    inkscape -z -e fig_mcfm_rung27.png -d 600 fig_mcfm_rung27.svg

_core.py defines the shared drawing helpers and the voxel renderer; layout.py
places the eight elements and writes the SVG. They share a namespace, so this
driver execs them in order rather than importing.

NOTE inkscape here is 0.92.5 — the export flags are `-e` / `-A`, NOT `-o`.
"""
import pathlib, sys

_H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_H))
ns = {'__file__': str(_H / 'make_figure.py'), '__name__': '__main__'}
for part in ('_core.py', 'layout.py'):
    exec(compile((_H / part).read_text(), part, 'exec'), ns)
