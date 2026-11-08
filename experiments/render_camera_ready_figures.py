"""Refine four camera-ready figures using unchanged archived observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from _bootstrap import ROOT
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from camera_ready_common import BASELINE_COMMIT, WIDTH_IN, MIN_LABEL_PT, save_figure
from egress_sim.figure_protocol import validate_frozen_artifact_bundle

NAMES = ("fig1_system_overview", "fig2_performance_landscape",
         "fig3_context_validation", "fig4_decision_diagnostics")
DEFAULT_OUTPUT = ROOT / "camera_ready_artifacts/2026-11-08"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): digest(p)
            for directory in (ROOT / "paper_artifacts", ROOT / "revision_artifacts")
            for p in sorted(directory.rglob("*")) if p.is_file()}


def validate_existing_inputs() -> None:
    validate_frozen_artifact_bundle(ROOT / "paper_artifacts")
    revision = ROOT / "revision_artifacts/2026-11"
    for line in (revision / "ARTIFACTS.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if digest(revision / name) != expected:
            raise ValueError(f"Archived revision artifact changed: {name}")


def font_audit(fig: plt.Figure) -> dict:
    fig.canvas.draw()
    width, height = fig.get_size_inches()
    if not np.isclose(width, WIDTH_IN, atol=1e-10, rtol=0):
        raise ValueError("Figure canvas does not match measured IEEE text width")
    texts = []
    bounds = fig.bbox
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            omitted = not ax.axison or not axis.get_visible()
            if omitted:
                hidden.update((axis.label, axis.offsetText))
            low, high = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if omitted or tick.get_loc() < low - 1e-9 or tick.get_loc() > high + 1e-9:
                    hidden.update((tick.label1, tick.label2))
    for artist in fig.findobj(match=Text):
        value = artist.get_text().strip()
        if artist in hidden or not artist.get_visible() or not value:
            continue
        size = float(artist.get_fontsize())
        # Mathematical script glyphs are generated internally by mathtext;
        # the parent label itself must satisfy the main-label size threshold.
        if size < MIN_LABEL_PT - 1e-7:
            raise ValueError(f"Figure text below {MIN_LABEL_PT} pt: {value!r}: {size}")
        box = artist.get_window_extent(fig.canvas.get_renderer())
        if box.x0 < bounds.x0 - .5 or box.y0 < bounds.y0 - .5 or box.x1 > bounds.x1 + .5 or box.y1 > bounds.y1 + .5:
            raise ValueError(f"Figure text extends outside the fixed canvas: {value!r}")
        texts.append({"text": value, "source_points": size,
                      "effective_points_at_textwidth": size})
    return {"width_inches": float(width), "height_inches": float(height),
            "pdf_width_points": float(width * 72), "latex_width_tex_points": 516,
            "placement": r"width=\textwidth", "all_text_inside_canvas": True,
            "min_main_label_points": min(t["source_points"] for t in texts),
            "text": texts}


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_manifest(output: Path) -> None:
    lines = [digest(p) + "  " + p.relative_to(output).as_posix()
             for p in sorted(output.rglob("*"))
             if p.is_file() and p.name != "ARTIFACTS.sha256"]
    (output / "ARTIFACTS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run(output: Path = DEFAULT_OUTPUT) -> None:
    from camera_ready_fig13 import figure1, figure3
    from camera_ready_fig2 import figure2
    from camera_ready_fig4 import figure4

    validate_existing_inputs()
    before = input_hashes()
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "before").mkdir(exist_ok=True)
    tables = ROOT / "paper_artifacts/tables"
    summary = pd.read_csv(ROOT / "revision_artifacts/2026-11/decision_diagnostics_summary.csv")
    data = {}; fonts = {}
    builders = (lambda: figure1(tables), lambda: figure2(tables),
                lambda: figure3(tables), lambda: figure4(summary))
    for name, build in zip(NAMES, builders):
        fig, values = build()
        fonts[name] = font_audit(fig)
        data[name] = values
        save_figure(fig, output / "figures" / name)
        shutil.copyfile(output / "figures" / (name + ".pdf"), ROOT / "paper" / (name + ".pdf"))
        original = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:paper/{name}.pdf"], cwd=ROOT)
        (output / "before" / (name + ".pdf")).write_bytes(original)
    if input_hashes() != before:
        raise AssertionError("Archived research artifacts were modified by figure rendering")
    metadata = {"baseline_commit": BASELINE_COMMIT, "scope": "presentation-only refinement",
                "data_and_statistics_changed": False, "new_experiments": 0,
                "input_sha256": before, "font_audit": fonts}
    (output / "figure_data.json").write_text(json.dumps(data, indent=2, default=_json) + "\n", encoding="utf-8", newline="\n")
    (output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json) + "\n", encoding="utf-8", newline="\n")
    write_manifest(output)
    print(f"Refined four figures at full IEEE text width in {output}")
    print("All original and prior-revision artifacts unchanged; no simulations executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
