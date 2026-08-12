"""
Stage 1 preprocessing: build paired (64mT ULF input, 3T target) slices.

For each subject that has both a 3T target (ses-c01, acq_HF) and a usable ULF
input, produce axial slices of:
  input  = [T1, T2, FLAIR]  (64mT / ULF)   + input_avail
  target = [T1, T2, FLAIR]  (3T)           + target_avail

Input source per dataset:
  - ulfenc : ULF Axial T1/T2/FLAIR, already on the 3T grid           -> pass through
  - webb   : ULF Axial T1/T2/FLAIR, already on the 3T grid           -> pass through
  - kcl    : ULF Axial/Coronal/Sagittal T1/T2 (no FLAIR)             -> register to 3T (SimpleITK)

kcl used to be read from its TomoBrain reconstruction. The data authors have since
flagged those as miscalculated, and TomoBrain is a single volume anyway, so it could
never produce the coronal/sagittal pairs this stage needs. Each plane now comes from
its own acquisition.

Target is the same 3T triplet as Stage 0 (kcl/webb registered, ulfenc native).
Missing modality slots are copy-filled; input_avail / target_avail record which
slots are real. kcl = [1,1,0] on both sides.

Output (mirrors Stage 0 layout, but each slice stores input+target):
  <out_root>/<dataset>/<subject>/slice_<zzz>.npz   # input[3,H,W], target[3,H,W] float16
  <out_root>/metadata.jsonl                        # + input_avail, target_avail
"""
import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F

MODS = ["T1w", "T2w", "FLAIR"]
PLANES = ["axial", "coronal", "sagittal"]

# identical geometry to Stage 0: same reslice convention, same pad-to-canvas, so a
# Stage 1 latent lines up channel-for-channel with the prior it is finetuning
sys.path.insert(0, str(Path(__file__).parent))
from register_and_preprocess_3t import take_slice, to_canvas  # noqa: E402
HF_RE = re.compile(r"^(sub-[^_]+)_ses-c01_acq[-_]HF_(T1w|T2w|FLAIR)\.nii(?:\.gz)?$")


# ------------------------------------------------------------------ discovery
def find_targets(medical_root, ds):
    """subject -> {mod: 3T path} for ses-c01 acq_HF."""
    out = defaultdict(dict)
    for anat in sorted((medical_root / ds).glob("*/ses-c01/anat")):
        for p in anat.glob("*HF*.nii.gz"):
            m = HF_RE.match(p.name)
            if m:
                out[m.group(1)][m.group(2)] = p
    return dict(out)


def find_inputs(medical_root, ds, sub, plane='axial'):
    """{mod: ULF path} for this subject, per-dataset source rule.

    webb: the raw ses-001/anat ULF T1/T2/FLAIR are NOT mutually registered (visible
    T1-vs-T2 skull offset). The authors' pipeline produced coregistered versions in
    derivatives/<sub>/<ses>/processing/coregistration/{T1,T2,FLAIR}_axi.nii.gz
    (T1->T2, FLAIR->T2, then ->HF-T2 grid), so use those. Falls back to raw if absent.
    kcl: TomoBrain ULF (already co-gridded with 3T). ulfenc: raw Axial (already aligned).
    """
    # The raw ses-00*/anat ULF is NOT registered to the 3T grid (it merely carries the
    # HF affine); the authors' pipeline wrote the registered versions under derivatives/
    # processing/coregistration. Use those.
    #   webb: {T1,T2,FLAIR}_axi.nii.gz   (Axial ULF -> HF-T2 space)
    #   kcl : {T1,T2}_TomoBrain.nii.gz   (TomoBrain ULF -> HF-T2 space; no FLAIR)
    if ds in ("webb", "kcl"):
        suffix = {"webb": ("axi", [("T1w", "T1"), ("T2w", "T2"), ("FLAIR", "FLAIR")]),
                  "kcl": ("TomoBrain", [("T1w", "T1"), ("T2w", "T2")])}[ds]
        tag, mods = suffix
        cor_dirs = sorted((medical_root / ds / "derivatives" / sub).glob("ses-*/processing/coregistration"))
        for cd in cor_dirs:                       # prefer the first session that has T1&T2
            got = {}
            for m, key in mods:
                f = cd / f"{key}_{tag}.nii.gz"
                if f.exists():
                    got[m] = f
            if "T1w" in got and "T2w" in got:
                return got
        # fall through to raw if no usable coregistration dir

    anat = list((medical_root / ds / sub).glob("ses-00*/anat"))
    if not anat:
        return {}
    files = [p for a in anat for p in a.glob("*.nii.gz")]
    # KCL: use the plane's own acquisition, NOT TomoBrain. The data authors flagged the
    # TomoBrain reconstructions as miscalculated, and KCL is the only site that acquired
    # ULF in all three planes anyway -- which is the whole point of the 3-plane rebuild.
    # ulfenc/webb only ever acquired Axial, so they are axial-only pairs.
    pat = plane.capitalize() if ds == "kcl" else "Axial"
    out = {}
    for m in MODS:
        cand = [p for p in files if pat in p.name and re.search(rf"_{m}\.nii", p.name)
                and re.search(r"acq[-_]ULF", p.name)]
        if cand:
            out[m] = sorted(cand)[0]
    return out


# ------------------------------------------------------------------ sitk / io
def load_sitk(p):
    return sitk.ReadImage(str(p), sitk.sitkFloat32)


def register_to(fixed, moving, interpolator=sitk.sitkBSpline):
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    tfm = reg.Execute(fixed, moving)
    res = sitk.Resample(moving, fixed, tfm, interpolator, 0.0, sitk.sitkFloat32)
    return res, float(reg.GetMetricValue())


def sitk_to_ras(img):
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    d = np.array(img.GetDirection()).reshape(3, 3)
    aff = np.eye(4); aff[:3, :3] = d * sp; aff[:3, 3] = org
    aff = np.diag([-1, -1, 1, 1]) @ aff
    return nib.as_closest_canonical(nib.Nifti1Image(arr, aff)).get_fdata(dtype=np.float32)


def load_ras(p):
    return nib.as_closest_canonical(nib.load(str(p))).get_fdata(dtype=np.float32)


def robust_norm01(v, eps=1e-8, min_nz=100):
    v = np.nan_to_num(v.astype(np.float32))
    mask = np.abs(v) > eps
    if int(mask.sum()) < min_nz:
        return None
    p01, p99 = np.percentile(v[mask], 1), np.percentile(v[mask], 99)
    if p99 <= p01 + 1e-6:
        return None
    return np.clip((np.clip(v, p01, p99) - p01) / (p99 - p01 + 1e-6), 0, 1).astype(np.float32)


def resize2d(s, res):
    t = torch.from_numpy(s)[None, None]
    return F.interpolate(t, (res, res), mode="bicubic", align_corners=False)[0, 0].clamp(0, 1).numpy()


# ------------------------------------------------------------------ per subject
def _aff_key(p):
    return np.round(nib.load(str(p)).affine, 1).tobytes()


def build_side(mods_paths, ref_paths, ref_fixed_sitk, qc, force_register=False):
    """Return (avail[3], {mod: normed RAS vol}) on the reference grid, copy-filled.

    webb and ulfenc ULF really do share the 3T grid, so they only need reorient +
    normalize, and registration stays a fallback for a mismatched affine.

    KCL's per-plane ULF is the exception and needs force_register: those volumes were
    resampled onto the 3T grid, so their affine MATCHES and the fallback never fires,
    but the subject's head sits at a different angle in each acquisition -- they were
    separate scans. Trusting the affine there produces pairs that look aligned in the
    header and are visibly rotated in the image, which is the one thing Stage 1 cannot
    tolerate: channel-concatenation assumes the same anatomy at the same pixel.
    """
    avail = [1 if m in mods_paths else 0 for m in MODS]
    if not (avail[0] and avail[1]):                # require T1 & T2
        return None, None
    ref_key = _aff_key(ref_paths["T2w"])
    vols = {}
    for m in MODS:
        if m not in mods_paths:
            continue
        if not force_register and _aff_key(mods_paths[m]) == ref_key:   # truly on the 3T grid
            v = robust_norm01(load_ras(mods_paths[m]))
        else:                                      # rigid + Mattes MI onto the 3T grid
            res, met = register_to(ref_fixed_sitk, load_sitk(mods_paths[m]))
            qc[m] = met
            v = robust_norm01(sitk_to_ras(res))
        if v is None:
            return None, None
        vols[m] = v
    ref_shape = vols["T2w"].shape
    for m in MODS:                                  # copy-fill missing (prefer T2 then T1)
        if m not in vols:
            vols[m] = vols["T2w"].copy()
        elif vols[m].shape != ref_shape:
            return None, None
    return avail, vols


def process_subject(ds, sub, tgt_paths, inp_paths, out_root, res, min_fg, meta_f, qc_f,
                    plane="axial"):
    if "T2w" not in tgt_paths or "T1w" not in tgt_paths:
        return 0, "no_target_T1T2"
    if "T2w" not in inp_paths or "T1w" not in inp_paths:
        return 0, "no_input_T1T2"

    # 3T T2w is the reference grid; everything in medical/ is already registered to
    # it, so build_side passes through (registration is a no-op fallback).
    fixed = load_sitk(tgt_paths["T2w"])
    qc_t, qc_i = {}, {}

    tgt_avail, tgt = build_side(tgt_paths, tgt_paths, fixed, qc_t)
    if tgt is None:
        return 0, "target_build_fail"

    inp_avail, inp = build_side(inp_paths, tgt_paths, fixed, qc_i,
                                force_register=(ds == "kcl"))
    if inp is None:
        return 0, "input_build_fail"

    if tgt["T2w"].shape != inp["T2w"].shape:
        return 0, "grid_mismatch"

    ti = np.stack([tgt[m] for m in MODS], 0)        # [3,X,Y,Z] canonical RAS
    ii = np.stack([inp[m] for m in MODS], 0)
    n_along = {"sagittal": ti.shape[1], "coronal": ti.shape[2], "axial": ti.shape[3]}[plane]
    sub_out = out_root / ds / sub
    sub_out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for k in range(n_along):
        ts, is_ = take_slice(ti, plane, k), take_slice(ii, plane, k)
        if (ts[:2] > 0.02).mean() < min_fg:
            continue
        tc = np.stack([to_canvas(ts[c], res) for c in range(3)], 0).astype(np.float16)
        ic = np.stack([to_canvas(is_[c], res) for c in range(3)], 0).astype(np.float16)
        fp = sub_out / f"{plane}_{k:03d}.npz"
        np.savez_compressed(fp, input=ic, target=tc)
        meta_f.write(json.dumps({
            "path": str(fp.relative_to(out_root)), "dataset": ds, "subject": sub, "slice": k,
            "orientation": plane,
            "input_avail": inp_avail, "target_avail": tgt_avail}) + "\n")
        kept += 1
    qc_f.write(json.dumps({"dataset": ds, "subject": sub, "plane": plane, "input_avail": inp_avail,
                           "target_avail": tgt_avail, "reg_target": qc_t, "reg_input": qc_i,
                           "slices": kept}) + "\n")
    return kept, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medical_root", default="/home/rintern14/ymk/medical")
    ap.add_argument("--out_root", default="/home/rintern14/ymk/data_stage1_pairs")
    ap.add_argument("--datasets", nargs="+", default=["kcl", "ulfenc", "webb"])
    ap.add_argument("--orientations", nargs="+", default=PLANES, choices=PLANES,
                    help="planes to build pairs for; only kcl has non-axial ULF")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--min_fg", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="subjects to skip, as 'sub-364' or 'webb/sub-364' "
                         "(e.g. incomplete data; re-run without this later)")
    args = ap.parse_args()
    # accept both "sub-364" and "webb/sub-364"
    excl = set(args.exclude) | {e.split("/")[-1] for e in args.exclude}

    medical_root, out_root = Path(args.medical_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_f = open(out_root / "metadata.jsonl", "w")
    qc_f = open(out_root / "pairs_qc.jsonl", "w")
    totals = defaultdict(int)
    done = 0
    for ds in args.datasets:
        targets = find_targets(medical_root, ds)
        for sub in sorted(targets):
            if sub in excl or f"{ds}/{sub}" in excl:
                totals["excluded"] += 1
                print(f"[{ds}] {sub}: excluded", flush=True)
                continue
            # kcl acquired ULF in all three planes; ulfenc and webb only axially, so
            # asking them for a coronal pair would just find no input volume.
            planes = args.orientations if ds == "kcl" else ["axial"]
            for plane in planes:
                inp = find_inputs(medical_root, ds, sub, plane)
                kept, status = process_subject(ds, sub, targets[sub], inp, out_root,
                                               args.res, args.min_fg, meta_f, qc_f, plane)
                totals[f"{status}"] += 1; totals["slices"] += kept
                print(f"[{ds}] {sub} {plane}: {status} kept={kept} in={sorted(inp)}", flush=True)
            done += 1
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break
    meta_f.close(); qc_f.close()
    print("SUMMARY:", dict(totals))


if __name__ == "__main__":
    main()
