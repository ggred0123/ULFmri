#!/usr/bin/env python3
"""Safely assemble the Stage 1 NIfTI tree from the downloaded ZIP bundles.

The download contains overlapping exports:

* ``archive-*.zip`` contains the original ``archive/kcl`` and ``archive/webb``.
* ``kcl-*.zip`` contains the complete, later KCL tree and replaces archive/KCL.
* ``ulfenc-*.zip`` contains the complete ULF-EnC tree.
* ``webb-*.zip`` is only a partial rerun, so the complete archive/Webb tree is used.

The output is normalized to ``OUT/{kcl,ulfenc,webb}``, which is the layout expected
by ``preprocess_pairs.py``.  ZIP members outside those selected trees are ignored.
"""
import argparse
import json
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


DATASETS = ("kcl", "ulfenc", "webb")


def archive_family(path: Path) -> str:
    name = path.name.lower()
    for family in ("archive", "kcl", "ulfenc", "webb", "code"):
        if name.startswith(family + "-"):
            return family
    return "unknown"


def destination_parts(family: str, member: str):
    """Map one ZIP member to its normalized output path, or return None."""
    parts = list(PurePosixPath(member).parts)
    if not parts or any(p in ("", ".", "..") for p in parts):
        return None

    # The complete KCL export supersedes archive/KCL.  The standalone Webb ZIP is
    # a partial rerun; the complete Webb tree is the one inside archive-*.zip.
    if family == "archive":
        if len(parts) >= 2 and parts[0] == "archive" and parts[1] == "webb":
            return tuple(parts[1:])
        return None
    if family == "kcl" and parts[0] == "kcl":
        return tuple(parts)
    if family == "ulfenc" and parts[0] == "ulfenc":
        return tuple(parts)
    return None


def build_plan(archive_root: Path):
    zips = sorted(archive_root.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no ZIP files found in {archive_root}")

    plan = []
    families = set()
    destinations = set()
    for zip_path in zips:
        family = archive_family(zip_path)
        if family not in {"archive", "kcl", "ulfenc"}:
            continue
        try:
            with ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    parts = destination_parts(family, info.filename)
                    if parts is None:
                        continue
                    rel = Path(*parts)
                    if rel in destinations:
                        raise RuntimeError(f"duplicate selected output path: {rel}")
                    destinations.add(rel)
                    families.add(family)
                    plan.append((zip_path, info.filename, rel, info.file_size,
                                 info.date_time, info.external_attr))
        except BadZipFile as exc:
            raise RuntimeError(f"invalid ZIP central directory: {zip_path}: {exc}") from exc

    missing = {"archive", "kcl", "ulfenc"} - families
    if missing:
        raise RuntimeError(
            "missing required archive families: " + ", ".join(sorted(missing))
        )
    return plan


def safe_output(root: Path, rel: Path) -> Path:
    dest = root.joinpath(rel)
    resolved_root = root.resolve()
    resolved_parent = dest.parent.resolve()
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise RuntimeError(f"unsafe ZIP output path: {rel}")
    return dest


def extract(plan, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    bytes_written = 0
    current_zip = None
    zf = None
    try:
        for index, (zip_path, member, rel, size, date_time, external_attr) in enumerate(plan, 1):
            if zip_path != current_zip:
                if zf is not None:
                    zf.close()
                zf = ZipFile(zip_path)
                current_zip = zip_path
                print(f"[{index}/{len(plan)}] {zip_path.name}", flush=True)

            dest = safe_output(out_root, rel)
            if dest.exists():
                if dest.is_file() and dest.stat().st_size == size:
                    skipped += 1
                    continue
                raise FileExistsError(
                    f"refusing to replace an existing file with different size: {dest}"
                )

            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".partial")
            if partial.exists():
                partial.unlink()
            with zf.open(member) as src, open(partial, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            if partial.stat().st_size != size:
                partial.unlink(missing_ok=True)
                raise IOError(f"size mismatch after extraction: {member}")
            os.replace(partial, dest)

            mode = (external_attr >> 16) & 0o777
            if mode:
                dest.chmod(stat.S_IMODE(mode))
            try:
                ts = datetime(*date_time).timestamp()
                os.utime(dest, (ts, ts))
            except (OSError, OverflowError, ValueError):
                pass
            written += 1
            bytes_written += size
            if written % 500 == 0:
                print(f"  extracted={written} skipped={skipped}", flush=True)
    finally:
        if zf is not None:
            zf.close()
    return written, skipped, bytes_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive_root", required=True,
                    help="directory containing archive-*.zip, kcl-*.zip and ulfenc-*.zip")
    ap.add_argument("--out_root", required=True,
                    help="normalized output root; creates kcl/, ulfenc/ and webb/")
    ap.add_argument("--dry_run", action="store_true",
                    help="inspect ZIP central directories and print the extraction plan only")
    args = ap.parse_args()

    archive_root = Path(args.archive_root)
    out_root = Path(args.out_root)
    if not archive_root.is_dir():
        ap.error(f"archive root is not a directory: {archive_root}")

    plan = build_plan(archive_root)
    total = sum(item[3] for item in plan)
    by_ds = {ds: sum(1 for item in plan if item[2].parts[0] == ds) for ds in DATASETS}
    print(f"selected {len(plan)} files ({total / 2**30:.2f} GiB): " +
          "  ".join(f"{ds}={by_ds[ds]}" for ds in DATASETS), flush=True)
    print("selection: named KCL + named ULF-EnC + complete archive/Webb", flush=True)
    if args.dry_run:
        return

    free = shutil.disk_usage(out_root.parent if out_root.parent.exists() else archive_root).free
    existing_bytes = sum(item[3] for item in plan
                         if (out_root / item[2]).is_file()
                         and (out_root / item[2]).stat().st_size == item[3])
    needed = total - existing_bytes
    if free < needed + 2 * 2**30:
        ap.error(
            f"insufficient free space: need about {needed / 2**30:.1f} GiB plus "
            f"2 GiB margin, have {free / 2**30:.1f} GiB"
        )

    written, skipped, bytes_written = extract(plan, out_root)
    marker = {
        "archive_root": str(archive_root.resolve()),
        "files_selected": len(plan),
        "files_written": written,
        "files_reused": skipped,
        "bytes_written": bytes_written,
        "selection": "named KCL + named ULF-EnC + complete archive/Webb",
    }
    with open(out_root / ".stage1_archive_extract.json", "w") as f:
        json.dump(marker, f, indent=2)
        f.write("\n")
    print(f"DONE -> {out_root}  written={written} reused={skipped}", flush=True)


if __name__ == "__main__":
    main()
