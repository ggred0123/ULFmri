import re
import argparse
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# LF NIfTI 파일명 파싱용: sub-XXX_ses-HFC_뭔가.nii.gz
LF_RE = re.compile(r"^(sub-[^_]+)_ses-(HFC|HFE)_(.+)\.nii(\.gz)?$")

# kcl_normalized 샘플 폴더명 파싱용: sub-XXX_ses-HFC_tomo (확장자 없음)
NORM_SAMPLE_RE = re.compile(r"^(sub-[^_]+)_ses-(HFC|HFE)_(.+)$")


def load_nii_to_float32(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)  # (X,Y,Z)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def robust_norm01_nonzero(vol: np.ndarray, eps: float = 1e-8, min_nonzero: int = 100) -> np.ndarray | None:
    vol = vol.astype(np.float32, copy=False)

    mask = np.abs(vol) > eps
    nz = int(mask.sum())
    if nz < min_nonzero:
        return None

    v = vol[mask]
    p01 = float(np.percentile(v, 1))
    p99 = float(np.percentile(v, 99))
    if p99 <= p01 + 1e-6:
        return None

    vol = np.clip(vol, p01, p99)
    vol = (vol - p01) / (p99 - p01 + 1e-6)
    vol = np.clip(vol, 0.0, 1.0)
    return vol.astype(np.float32, copy=False)


def save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr.astype(np.float32, copy=False))


def make_out_tag(tag: str, direction: str) -> str:
    """
    tag 안에 direction 정보가 이미 있으면 그대로 사용.
    없으면 충돌 방지 위해 _{direction} 붙임.
    """
    t = tag.lower()
    d = direction.lower()
    if t == d or t.endswith(f"_{d}") or t.startswith(f"{d}_") or f"_{d}_" in t:
        return tag
    return f"{tag}_{direction}"


def build_hf_map_from_kcl_normalized(kcl_norm_root: Path, subset: str) -> dict[tuple[str, str], Path]:
    """
    kcl_normalized/<subset>/<sample_dir>/HF_T1.npy 를 스캔해서
    (sub, ses) -> HF_T1.npy 경로 매핑을 만든다.
    """
    subset_dir = kcl_norm_root / subset
    if not subset_dir.is_dir():
        raise ValueError(f"Missing subset folder: {subset_dir}")

    hf_map: dict[tuple[str, str], Path] = {}
    missing_hf = 0

    for sd in sorted([p for p in subset_dir.iterdir() if p.is_dir()]):
        m = NORM_SAMPLE_RE.match(sd.name)
        if m is None:
            continue

        sub, ses = m.group(1), m.group(2)
        hf_path = sd / "HF_T1.npy"
        if not hf_path.exists():
            missing_hf += 1
            continue

        # 같은 (sub,ses)가 여러 번 나와도 HF는 같을 가능성이 크니 첫 번째를 채택
        hf_map.setdefault((sub, ses), hf_path)

    if len(hf_map) == 0:
        raise ValueError(f"No HF_T1.npy found under: {subset_dir}")

    if missing_hf > 0:
        print(f"[WARN] {subset}: {missing_hf} sample dirs missing HF_T1.npy")

    return hf_map


def convert_subset_by_directions(
    lf_root: Path,
    kcl_norm_root: Path,
    out_root: Path,
    subset: str,
    directions: list[str],
    overwrite: bool = False,
    eps: float = 1e-8,
    min_nonzero: int = 100,
    require_same_shape: bool = False,
    copy_hf: bool = False,
):
    # (sub,ses) -> HF npy path (split 로직은 여기서 결정됨)
    hf_map = build_hf_map_from_kcl_normalized(kcl_norm_root, subset)
    keys_in_subset = set(hf_map.keys())

    ok, skip = 0, 0
    skip_reason = defaultdict(int)

    print(f"\n[{subset}] keys(from kcl_normalized) = {len(keys_in_subset)} subjects/sessions")

    for d in directions:
        t1_dir = lf_root / f"T1_{d}"
        t2_dir = lf_root / f"T2_{d}"

        if not t1_dir.is_dir():
            print(f"[WARN] Missing: {t1_dir} (skip direction={d})")
            continue
        if not t2_dir.is_dir():
            print(f"[WARN] Missing: {t2_dir} (skip direction={d})")
            continue

        t1_files = sorted([p for p in t1_dir.glob("*.nii*") if p.is_file()])
        print(f"[{subset}] direction={d}: found {len(t1_files)} T1 files")

        for t1_path in tqdm(t1_files, desc=f"{subset}-{d}"):
            m = LF_RE.match(t1_path.name)
            if m is None:
                skip += 1
                skip_reason["lf_name_regex_mismatch"] += 1
                continue

            sub, ses, tag = m.group(1), m.group(2), m.group(3)

            if (sub, ses) not in keys_in_subset:
                # split에 없는 subject/session이면 건너뜀
                continue

            t2_path = t2_dir / t1_path.name
            if not t2_path.exists():
                skip += 1
                skip_reason[f"missing_t2_{d}"] += 1
                continue

            hf_path = hf_map[(sub, ses)]
            out_tag = make_out_tag(tag, d)

            out_sample_dir = out_root / subset / f"{sub}_ses-{ses}_{out_tag}"
            out_ulf_t1 = out_sample_dir / "ULF_T1.npy"
            out_ulf_t2 = out_sample_dir / "ULF_T2.npy"
            out_hf_t1  = out_sample_dir / "HF_T1.npy"

            if (not overwrite) and out_ulf_t1.exists() and out_ulf_t2.exists() and out_hf_t1.exists():
                ok += 1
                continue

            try:
                a = load_nii_to_float32(t1_path)
                b = load_nii_to_float32(t2_path)

                if a.ndim != 3 or b.ndim != 3:
                    skip += 1
                    skip_reason["not_3d_lf"] += 1
                    continue

                a_n = robust_norm01_nonzero(a, eps=eps, min_nonzero=min_nonzero)
                if a_n is None:
                    skip += 1
                    skip_reason["t1_too_empty_or_constant"] += 1
                    continue

                b_n = robust_norm01_nonzero(b, eps=eps, min_nonzero=min_nonzero)
                if b_n is None:
                    skip += 1
                    skip_reason["t2_too_empty_or_constant"] += 1
                    continue

                # HF는 kcl_normalized의 결과를 그대로 가져오되 안전하게 float32 + NaN 처리
                if copy_hf:
                    out_sample_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(hf_path, out_hf_t1)
                    hf_arr = np.load(str(out_hf_t1)).astype(np.float32, copy=False)
                else:
                    hf_arr = np.load(str(hf_path)).astype(np.float32, copy=False)
                    hf_arr = np.nan_to_num(hf_arr, nan=0.0, posinf=0.0, neginf=0.0)

                if require_same_shape:
                    if a_n.shape != b_n.shape or a_n.shape != hf_arr.shape:
                        skip += 1
                        skip_reason["shape_mismatch"] += 1
                        continue

                save_npy(out_ulf_t1, a_n)
                save_npy(out_ulf_t2, b_n)
                if not copy_hf:
                    save_npy(out_hf_t1, hf_arr)

                ok += 1

            except Exception as e:
                print(f"[ERROR] {subset} dir={d} file={t1_path.name}: {e}")
                skip += 1
                skip_reason["exception"] += 1

    print(f"[{subset}] done. ok={ok}, skipped={skip}")
    if skip > 0:
        print(f"[{subset}] skip reasons:")
        for k, v in sorted(skip_reason.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lf_root", type=str, default="/home/rintern14/ymk/dataset_kcl",
                        help="LF dataset root (contains T1_axi/T2_axi/T1_cor/T2_cor/...)")
    parser.add_argument("--kcl_norm_root", type=str, default="/home/rintern14/ymk/kcl_normalized",
                        help="Existing kcl_normalized root (contains Train/Valid/Test and HF_T1.npy inside)")
    parser.add_argument("--out_root", type=str, default="/home/rintern14/ymk/dataset_kcl_by_direction",
                        help="Output root for new direction-based dataset")
    parser.add_argument("--subsets", nargs="+", default=["Train", "Valid", "Test"],
                        help="Subset folder names to process (default: Train Valid Test)")
    parser.add_argument("--dirs", nargs="+", default=["axi", "cor", "sag"],
                        help="Directions to include (default: axi cor sag)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min_nonzero", type=int, default=100)
    parser.add_argument("--require_same_shape", action="store_true")
    parser.add_argument("--copy_hf", action="store_true",
                        help="If set, copy HF_T1.npy file directly (faster). Otherwise load+save.")
    args = parser.parse_args()

    lf_root = Path(args.lf_root)
    kcl_norm_root = Path(args.kcl_norm_root)
    out_root = Path(args.out_root)

    for subset in args.subsets:
        convert_subset_by_directions(
            lf_root=lf_root,
            kcl_norm_root=kcl_norm_root,
            out_root=out_root,
            subset=subset,
            directions=args.dirs,
            overwrite=args.overwrite,
            eps=args.eps,
            min_nonzero=args.min_nonzero,
            require_same_shape=args.require_same_shape,
            copy_hf=args.copy_hf,
        )


if __name__ == "__main__":
    main()
