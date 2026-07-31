"""
Stage 0 preprocessing: build the 3T unconditional prior dataset.

Walks the BIDS-style `medical/{kcl,ulfenc,webb}` tree, collects every subject
that has a 3T target session (`ses-c01`, `acq_HF`), and turns each 3T volume
into per-slice 3-channel [T1w, T2w, FLAIR] images.

Design (combined / pseudo-RGB, per the training plan):
  - channels are always [T1w, T2w, FLAIR]
  - a missing modality slot is filled by copying an available modality
    (KCL has no FLAIR -> FLAIR := T2w, target_avail = [1, 1, 0])
  - target_avail is stored as metadata (used later as a conditioning signal,
    NOT as a latent-space loss mask -- copies keep the target valid everywhere)

Only intra-subject *aligned* datasets are handled here (kcl, ulfenc). webb's
3T modalities are not co-registered and are deferred to a registration step.

Output:
  <out_root>/<dataset>/<subject>/slice_<zzz>.npy   # float32 [3, H, W] in [0, 1]
  <out_root>/metadata.jsonl                        # one row per slice
"""
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch.nn.functional as F  # only for a clean bilinear resize
import torch

MODALITIES = ["T1w", "T2w", "FLAIR"]
# HF (3T) target files: sub-XXX_ses-c01_acq[-_]HF_<MOD>.nii.gz
HF_RE = re.compile(r"^(sub-[^_]+)_ses-c01_acq[-_]HF_(T1w|T2w|FLAIR)\.nii(?:\.gz)?$")


def robust_norm01(vol: np.ndarray, eps: float = 1e-8, min_nonzero: int = 100):
    """1-99 percentile clip on non-zero voxels, scaled to [0, 1]. None if degenerate."""
    vol = np.nan_to_num(vol.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(vol) > eps
    if int(mask.sum()) < min_nonzero:
        return None
    v = vol[mask]
    p01, p99 = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    if p99 <= p01 + 1e-6:
        return None
    vol = np.clip(vol, p01, p99)
    return ((vol - p01) / (p99 - p01 + 1e-6)).astype(np.float32)


def load_canonical(path: Path) -> np.ndarray:
    """Load a NIfTI, reorient to closest-canonical RAS, return float32 array."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    return img.get_fdata(dtype=np.float32)


def discover_subjects(medical_root: Path, datasets):
    """dataset -> {subject -> {mod -> path}} for 3T (ses-c01 acq_HF) volumes."""
    found = {}
    for ds in datasets:
        subj = defaultdict(dict)
        for anat in sorted((medical_root / ds).glob("*/ses-c01/anat")):
            for p in anat.glob("*HF*.nii.gz"):
                m = HF_RE.match(p.name)
                if m:
                    subj[m.group(1)][m.group(2)] = p
        found[ds] = dict(subj)
    return found


def resize2d(slc: np.ndarray, res: int) -> np.ndarray:
    t = torch.from_numpy(slc)[None, None]
    t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float32)


def build_channels(mods: dict):
    """Return (avail[3], {mod: vol}) with missing modalities copy-filled.

    Copy rule: fill each missing slot with a random-choice-free deterministic
    copy of an available modality (T2w preferred, else T1w).
    """
    avail = [1 if m in mods else 0 for m in MODALITIES]
    if not any(avail):
        return None, None
    vols = {}
    for m in MODALITIES:
        if m in mods:
            v = robust_norm01(load_canonical(mods[m]))
            if v is None:
                return None, None
            vols[m] = v
    if not vols:
        return None, None
    # deterministic donor for copy-fill: prefer T2w, then T1w, then whatever exists
    donor = next((m for m in ["T2w", "T1w", "FLAIR"] if m in vols), None)
    ref_shape = vols[donor].shape
    for m in MODALITIES:
        if m not in vols:
            vols[m] = vols[donor].copy()
        elif vols[m].shape != ref_shape:
            return None, None  # intra-subject misalignment -> skip (e.g. webb)
    return avail, vols


def process(args):
    medical_root = Path(args.medical_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    found = discover_subjects(medical_root, args.datasets)
    meta_f = open(out_root / "metadata.jsonl", "w")
    n_slices = 0
    n_subj = 0
    skipped = defaultdict(int)

    for ds, subjects in found.items():
        for sub, mods in sorted(subjects.items()):
            avail, vols = build_channels(mods)
            if vols is None:
                skipped[f"{ds}:build_fail"] += 1
                continue
            # stack [3, X, Y, Z] canonical RAS; axial slices along axis 2 (I->S)
            stack = np.stack([vols[m] for m in MODALITIES], axis=0)
            X, Y, Z = stack.shape[1:]
            sub_out = out_root / ds / sub
            sub_out.mkdir(parents=True, exist_ok=True)
            kept = 0
            for z in range(Z):
                slc = stack[:, :, :, z]  # [3, X, Y]
                # foreground filter on T1w/T2w channels
                fg = (slc[:2] > 0.02).mean()
                if fg < args.min_fg:
                    continue
                chans = np.stack([resize2d(slc[c], args.res) for c in range(3)], axis=0)
                fp = sub_out / f"slice_{z:03d}.npy"
                np.save(fp, chans.astype(np.float32))
                meta_f.write(json.dumps({
                    "path": str(fp.relative_to(out_root)),
                    "dataset": ds, "subject": sub, "slice": z,
                    "target_avail": avail,
                }) + "\n")
                kept += 1
                n_slices += 1
            if kept:
                n_subj += 1
            print(f"[{ds}] {sub}: avail={avail} shape={(X,Y,Z)} kept={kept}", flush=True)

    meta_f.close()
    print(f"\nDONE  subjects={n_subj}  slices={n_slices}  out={out_root}")
    if skipped:
        print("skipped:", dict(skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medical_root", default="/home/rintern14/ymk/medical")
    ap.add_argument("--out_root", default="/home/rintern14/ymk/data_stage0_3t")
    ap.add_argument("--datasets", nargs="+", default=["kcl", "ulfenc"],
                    help="aligned datasets only; webb needs registration first")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--min_fg", type=float, default=0.05,
                    help="min fraction of foreground voxels to keep a slice")
    args = ap.parse_args()
    process(args)


if __name__ == "__main__":
    main()
