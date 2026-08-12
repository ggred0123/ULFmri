"""
Stage 0 preprocessing WITH intra-subject registration.

Unlike preprocess_3t.py (which assumed modalities already share a grid), this
co-registers each subject's 3T modalities to a common reference (T2w) using
SimpleITK rigid + Mattes mutual-information registration, then slices.

Why: audit showed only ulfenc is affine-aligned (49/50). KCL 3T T1w/T2w share
a shape but are offset ~4mm; webb 3T modalities have different shapes entirely.
Rigid MI registration aligns them robustly (works whether the offset is a real
header translation or subject motion). Already-aligned subjects converge to ~identity.

Channels are [T1w, T2w, FLAIR]; reference = T2w.
  - require T1w AND T2w (else the subject is skipped / excluded)
  - FLAIR optional -> missing slot copy-filled from T2w, target_avail marks it 0

Slices are cut in all three orientations (--orientations, default axial+coronal
+sagittal). The 3T volumes are 3D, so coronal/sagittal are reslices of the same
registered volume, not separate acquisitions -- the prior gets 3x the slices and
learns all three viewing planes. The orientation is written to metadata so the
trainer can hand it to the text encoder as the prompt.

Output:
  <out_root>/<dataset>/<subject>/<orientation>_<kkk>.npy  # float32 [3, res, res] in [0,1]
  <out_root>/metadata.jsonl
  <out_root>/registration_qc.jsonl                        # per-subject metric / status
"""
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F

MODALITIES = ["T1w", "T2w", "FLAIR"]
HF_RE = re.compile(r"^(sub-[^_]+)_ses-c01_acq[-_]HF_(T1w|T2w|FLAIR)\.nii(?:\.gz)?$")
ORIENTATIONS = ["axial", "coronal", "sagittal"]


# ---------------------------------------------------------------- discovery
def discover(medical_root: Path, datasets):
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


# ---------------------------------------------------------------- registration
def load_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def register_rigid(fixed: sitk.Image, moving: sitk.Image, interpolator=sitk.sitkBSpline):
    """Rigid (Euler3D) + Mattes MI, 3-level multi-resolution. Returns (resampled, metric).

    `interpolator` applies to the FINAL resample, which is what determines output
    sharpness. Linear interpolation visibly softens the moving image (T1w/FLAIR)
    relative to the untouched fixed reference (T2w), so use B-spline by default.
    The optimizer's own interpolator stays linear -- it only affects the metric.
    """
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    tfm = reg.Execute(fixed, moving)
    metric = reg.GetMetricValue()
    resampled = sitk.Resample(moving, fixed, tfm, interpolator, 0.0, sitk.sitkFloat32)
    return resampled, float(metric)


def sitk_to_np_ras(img: sitk.Image) -> np.ndarray:
    """SimpleITK image -> numpy array reoriented to canonical RAS (X,Y,Z)."""
    # sitk GetArray is (z,y,x); convert via nibabel using the affine
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    arr = np.transpose(arr, (2, 1, 0))  # (x,y,z)
    # build affine from sitk metadata for canonical reorientation
    spacing = np.array(img.GetSpacing())
    origin = np.array(img.GetOrigin())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    aff = np.eye(4)
    aff[:3, :3] = direction * spacing
    aff[:3, 3] = origin
    # LPS (sitk) -> RAS (nibabel): flip x,y
    lps2ras = np.diag([-1, -1, 1, 1])
    aff = lps2ras @ aff
    nii = nib.Nifti1Image(arr, aff)
    return nib.as_closest_canonical(nii).get_fdata(dtype=np.float32)


# ---------------------------------------------------------------- intensity / slicing
def robust_norm01(vol, eps=1e-8, min_nonzero=100):
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


def resize2d(slc, res, mode="bicubic"):
    """Bicubic keeps more high-frequency detail than bilinear; it can overshoot
    slightly outside [0,1], so clamp afterwards."""
    t = torch.from_numpy(slc)[None, None]
    t = F.interpolate(t, size=(res, res), mode=mode, align_corners=False)
    return t[0, 0].clamp(0.0, 1.0).numpy().astype(np.float32)


def pad_center(a, H, W):
    out = np.zeros((H, W), dtype=np.float32)
    top, left = (H - a.shape[0]) // 2, (W - a.shape[1]) // 2
    out[top:top + a.shape[0], left:left + a.shape[1]] = a
    return out


def to_canvas(slc, res):
    """Center the slice on a res x res canvas WITHOUT changing its aspect ratio.

    Every 3T volume here is 1mm isotropic and <= 256 voxels along every axis, so at
    res=256 this is pure zero-padding: no interpolation at all, 1 pixel stays 1 mm,
    and a head is the same physical size in axial, coronal and sagittal. The old
    behaviour (stretch each slice to res x res) squashed coronal/sagittal by up to
    1.4x on ulfenc/webb -- an orientation-dependent geometric distortion the prior
    would have had to absorb. Slices bigger than the canvas are square-padded first
    and only then resampled, so the aspect ratio survives that path too.
    """
    H, W = slc.shape
    if max(H, W) > res:
        S = max(H, W)
        return resize2d(pad_center(slc, S, S), res)
    return pad_center(slc, res, res)


def take_slice(stack, orientation, k):
    """One slice out of a canonical-RAS [3, X, Y, Z] stack (X: L->R, Y: P->A, Z: I->S).

    Axial keeps the existing [X, Y] convention. Coronal/sagittal are transposed and
    flipped so that superior points up in the saved image -- without that they come
    out lying on their side, and every orientation would need its own upright prior.
    """
    if orientation == "axial":
        return np.ascontiguousarray(stack[:, :, :, k])          # [3, X, Y]
    s = stack[:, :, k, :] if orientation == "coronal" else stack[:, k, :, :]
    return np.ascontiguousarray(s.transpose(0, 2, 1)[:, ::-1, :])   # [3, Z(S up), X or Y]


# ---------------------------------------------------------------- per subject
def _affine_key(path):
    return np.round(nib.load(str(path)).affine, 1).tobytes()


def process_subject(ds, sub, mods, out_root, res, min_fg, qc_f, meta_f, orientations):
    if "T2w" not in mods or "T1w" not in mods:
        return 0, "skip_missing_T1_or_T2"

    # Leave already-aligned subjects untouched: if every present modality shares
    # the T2w affine, pass through (no registration). Only misaligned subjects
    # (kcl, webb) actually go through SimpleITK.
    ref_key = _affine_key(mods["T2w"])
    already_aligned = all(_affine_key(p) == ref_key for p in mods.values())

    fixed = load_sitk(mods["T2w"])                       # reference grid
    reg_vols = {"T2w": sitk_to_np_ras(fixed)}
    metrics = {}
    for m in ["T1w", "FLAIR"]:
        if m in mods:
            if already_aligned:
                reg_vols[m] = sitk_to_np_ras(load_sitk(mods[m]))
                metrics[m] = "passthrough"
            else:
                resampled, metric = register_rigid(fixed, load_sitk(mods[m]))
                reg_vols[m] = sitk_to_np_ras(resampled)
                metrics[m] = metric

    # normalize each real modality
    norm = {}
    for m, v in reg_vols.items():
        n = robust_norm01(v)
        if n is None:
            return 0, f"degenerate_{m}"
        norm[m] = n

    avail = [1 if m in norm else 0 for m in MODALITIES]  # FLAIR may be 0
    donor = "T2w"
    ref_shape = norm["T2w"].shape
    chans = {}
    for m in MODALITIES:
        chans[m] = norm[m] if m in norm else norm[donor].copy()
        if chans[m].shape != ref_shape:
            return 0, f"shape_mismatch_{m}"

    stack = np.stack([chans[m] for m in MODALITIES], 0)  # [3, X, Y, Z] RAS
    # axis of `stack` that each orientation steps along (0 is the modality axis)
    n_along = {"sagittal": stack.shape[1], "coronal": stack.shape[2], "axial": stack.shape[3]}
    sub_out = out_root / ds / sub
    sub_out.mkdir(parents=True, exist_ok=True)
    kept = 0
    per_ori = {}
    for ori in orientations:
        kept_ori = 0
        for k in range(n_along[ori]):
            slc = take_slice(stack, ori, k)
            # foreground fraction on the raw slice (before padding), so the threshold
            # means the same thing whatever the in-plane matrix is
            if (slc[:2] > 0.02).mean() < min_fg:
                continue
            ch = np.stack([to_canvas(slc[c], res) for c in range(3)], 0)
            fp = sub_out / f"{ori}_{k:03d}.npy"
            np.save(fp, ch.astype(np.float32))
            meta_f.write(json.dumps({
                "path": str(fp.relative_to(out_root)), "dataset": ds,
                "subject": sub, "slice": k, "orientation": ori, "target_avail": avail,
            }) + "\n")
            kept_ori += 1
        per_ori[ori] = kept_ori
        kept += kept_ori
    qc_f.write(json.dumps({
        "dataset": ds, "subject": sub, "avail": avail, "shape": list(stack.shape[1:]),
        "reg_metric": metrics, "slices": kept, "slices_per_orientation": per_ori,
    }) + "\n")
    return kept, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medical_root", default="/home/rintern14/ymk/medical")
    ap.add_argument("--out_root", default="/home/rintern14/ymk/data_stage0_3t_reg")
    ap.add_argument("--datasets", nargs="+", default=["kcl", "ulfenc", "webb"])
    ap.add_argument("--orientations", nargs="+", default=ORIENTATIONS, choices=ORIENTATIONS,
                    help="reslice planes to cut from each 3T volume")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--min_fg", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="process at most N subjects (0=all) for a smoke test")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    found = discover(Path(args.medical_root), args.datasets)

    meta_f = open(out_root / "metadata.jsonl", "w")
    qc_f = open(out_root / "registration_qc.jsonl", "w")
    totals = defaultdict(int)
    done = 0
    for ds, subjects in found.items():
        for sub, mods in sorted(subjects.items()):
            kept, status = process_subject(ds, sub, mods, out_root, args.res, args.min_fg,
                                           qc_f, meta_f, args.orientations)
            totals[status] += 1
            totals["slices"] += kept
            print(f"[{ds}] {sub}: {status} kept={kept} "
                  f"mods={sorted(mods)}", flush=True)
            done += 1
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break
    meta_f.close(); qc_f.close()
    print("\nSUMMARY:", dict(totals))


if __name__ == "__main__":
    main()
