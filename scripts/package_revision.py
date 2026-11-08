"""Build local review/Overleaf/reproduction deliverables from the checked repo."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / "deliverables"
STAMP = "2026-11-08"


def archive(path: Path, files: list[tuple[str, bytes]], stamp: tuple) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, content in sorted(files):
            info = zipfile.ZipInfo(name, stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content)
    with zipfile.ZipFile(path) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise AssertionError(f"Corrupt ZIP member: {bad}")


def git_blobs() -> dict[str, bytes]:
    """Read canonical blobs directly; checkout/archive can convert CRLF."""
    entries = []
    for entry in subprocess.check_output(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=ROOT).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        _, kind, sha = header.split()
        if kind != b"blob":
            raise ValueError(f"Unsupported export entry: {path!r}")
        entries.append((path.decode("utf-8"), sha.decode("ascii")))
    objects = list(dict.fromkeys(sha for _, sha in entries))
    batch = subprocess.check_output(["git", "cat-file", "--batch"], cwd=ROOT,
                                   input=("\n".join(objects) + "\n").encode("ascii"))
    content = {}; position = 0
    for expected in objects:
        end = batch.index(b"\n", position)
        sha, kind, size = batch[position:end].split()
        length = int(size); position = end + 1
        if sha.decode("ascii") != expected or kind != b"blob":
            raise ValueError("Unexpected Git object in export")
        content[expected] = batch[position:position + length]
        position += length + 1
    return {path: content[sha] for path, sha in entries}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    blobs = git_blobs()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    previous_path = DESTINATION / "DELIVERY_MANIFEST.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("source_commit") != head:
            archived = DESTINATION / "archive" / previous["source_commit"][:7]
            archived.mkdir(parents=True, exist_ok=True)
            for path in DESTINATION.iterdir():
                if path.is_file() and not (archived / path.name).exists():
                    shutil.copy2(path, archived / path.name)
    timestamp = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=ROOT))
    zip_stamp = datetime.fromtimestamp(timestamp, timezone.utc).timetuple()[:6]
    copies = {
        "paper/main.pdf": f"PIC2026_Revised_Manuscript_{STAMP}.pdf",
        "paper/response_to_reviewers.pdf": f"PIC2026_Response_to_Reviewer_{STAMP}.pdf",
        "paper/revision_notes_zh.md": f"PIC2026_Revision_Notes_zh_{STAMP}.md",
        "docs/pic2026_revision/QA_REPORT.md": f"PIC2026_QA_Report_{STAMP}.md",
    }
    optional = {
        "camera_ready_artifacts/2026-11-08/figure_comparison.pdf": f"PIC2026_Figure_Comparison_{STAMP}.pdf",
        "docs/pic2026_revision/FIGURE_REFINEMENT_QA.md": f"PIC2026_Figure_Refinement_QA_{STAMP}.md",
    }
    copies.update({source: name for source, name in optional.items() if source in blobs})
    for source, name in copies.items():
        (DESTINATION / name).write_bytes(blobs[source])
    overleaf_files = [(name.removeprefix("paper/"), data) for name, data in blobs.items() if name.startswith("paper/")]
    archive(DESTINATION / f"PIC2026_Overleaf_{STAMP}.zip", overleaf_files, zip_stamp)
    archive(DESTINATION / f"PIC2026_Reproducibility_{STAMP}.zip", list(blobs.items()), zip_stamp)
    records = {"source_commit": head, "zip_export": "canonical raw Git blobs", "files": {}}
    for name in [*copies.values(), f"PIC2026_Overleaf_{STAMP}.zip", f"PIC2026_Reproducibility_{STAMP}.zip"]:
        path = DESTINATION / name
        records["files"][name] = {"bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (DESTINATION / "DELIVERY_MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
