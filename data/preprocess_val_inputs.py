"""
Build an INPUT-ONLY validation set from unpaired ULF scans (no 3T target).

Use case: the real deployment scenario -- a low-field scan with no ground-truth
3T. webb has 53 subjects with ULF Axial (T1/T2/FLAIR) but no ses-c01 target.
These are never seen in Stage 1 training (which only uses subjects that HAVE a
target), so they are a genuine generalization check.

Output is written in the SAME layout Stage 0's precompute_latents.py consumes, so
you can encode it to 48ch latents with that script unchanged:
  <out_root>/<dataset>/<subject>/slice_<zzz>.npy   # [3,H,W] float [0,1]
  <out_root>/metadata.jsonl                        # target_avail carries input_avail
"""
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

from preprocess_pairs import MODS, find_inputs, load_ras, robust_norm01, resize2d


def find_all_ulf_subjects(medical_root, ds):
    return sorted(set(re.match(r".*/(sub-[^/]+)/", str(p)).group(1)
                      for p in (medical_root / ds).glob("*/ses-00*/anat/*ULF*.nii.gz")))


def has_target(medical_root, ds, sub):
    return any((medical_root / ds / sub).glob("ses-c01/anat/*HF*.nii.gz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medical_root", default="/home/rintern14/ymk/medical")
    ap.add_argument("--out_root", default="/home/rintern14/ymk/data_stage1_val_inputs")
    ap.add_argument("--datasets", nargs="+", default=["webb"])
    ap.add_argument("--unpaired_only", action="store_true", default=True,
                    help="only subjects WITHOUT a 3T target (true validation)")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--min_fg", type=float, default=0.05)
    ap.add_argument("--max_subjects", type=int, default=8, help="cap subjects (validation set is small)")
    args = ap.parse_args()

    medical_root, out_root = Path(args.medical_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_f = open(out_root / "metadata.jsonl", "w")
    totals = defaultdict(int)

    for ds in args.datasets:
        subs = find_all_ulf_subjects(medical_root, ds)
        picked = 0
        for sub in subs:
            if args.unpaired_only and has_target(medical_root, ds, sub):
                continue
            inp = find_inputs(medical_root, ds, sub)          # ULF Axial for webb
            if "T1w" not in inp or "T2w" not in inp:
                totals["skip_no_T1T2"] += 1
                continue
            avail = [1 if m in inp else 0 for m in MODS]
            vols = {m: robust_norm01(load_ras(inp[m])) for m in MODS if m in inp}
            if any(v is None for v in vols.values()):
                totals["degenerate"] += 1
                continue
            ref = vols["T2w"].shape
            for m in MODS:
                if m not in vols:
                    vols[m] = vols["T2w"].copy()
                elif vols[m].shape != ref:
                    vols = None
                    break
            if vols is None:
                totals["shape_mismatch"] += 1
                continue
            stack = np.stack([vols[m] for m in MODS], 0)
            Z = stack.shape[3]
            sub_out = out_root / ds / sub
            sub_out.mkdir(parents=True, exist_ok=True)
            kept = 0
            for z in range(Z):
                s = stack[:, :, :, z]
                if (s[:2] > 0.02).mean() < args.min_fg:
                    continue
                ch = np.stack([resize2d(s[c], args.res) for c in range(3)], 0).astype(np.float32)
                fp = sub_out / f"slice_{z:03d}.npy"
                np.save(fp, ch)
                meta_f.write(json.dumps({
                    "path": str(fp.relative_to(out_root)), "dataset": ds, "subject": sub,
                    "slice": z, "target_avail": avail}) + "\n")   # target_avail := input_avail
                kept += 1
            totals["ok"] += 1; totals["slices"] += kept
            print(f"[{ds}] {sub}: kept={kept} avail={avail}", flush=True)
            picked += 1
            if picked >= args.max_subjects:
                break
    meta_f.close()
    print("SUMMARY:", dict(totals))


if __name__ == "__main__":
    main()
