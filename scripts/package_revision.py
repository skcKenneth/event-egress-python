"""Build local review/Overleaf/reproduction deliverables from the checked repo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / "deliverables"
STAMP = "2026-11-08"


def archive(path: Path, files: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for source, name in sorted(files, key=lambda item: item[1]):
            bundle.write(source, name)
    with zipfile.ZipFile(path) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise AssertionError(f"Corrupt ZIP member: {bad}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    copies = {
        "paper/main.pdf": f"PIC2026_Revised_Manuscript_{STAMP}.pdf",
        "paper/response_to_reviewers.pdf": f"PIC2026_Response_to_Reviewer_{STAMP}.pdf",
        "paper/revision_notes_zh.md": f"PIC2026_Revision_Notes_zh_{STAMP}.md",
        "docs/pic2026_revision/QA_REPORT.md": f"PIC2026_QA_Report_{STAMP}.md",
    }
    for source, name in copies.items():
        shutil.copyfile(ROOT / source, DESTINATION / name)
    paper = ROOT / "paper"
    allowed = {".tex", ".bib", ".pdf", ".csv", ".md"}
    overleaf_files = [(p, p.relative_to(paper).as_posix()) for p in paper.rglob("*")
                      if p.is_file() and p.suffix in allowed]
    archive(DESTINATION / f"PIC2026_Overleaf_{STAMP}.zip", overleaf_files)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = [(ROOT / name, name) for name in tracked if name and (ROOT / name).is_file()]
    archive(DESTINATION / f"PIC2026_Reproducibility_{STAMP}.zip", files)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records = {"source_commit": head, "files": {}}
    for name in [*copies.values(), f"PIC2026_Overleaf_{STAMP}.zip", f"PIC2026_Reproducibility_{STAMP}.zip"]:
        path = DESTINATION / name
        records["files"][name] = {"bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (DESTINATION / "DELIVERY_MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
