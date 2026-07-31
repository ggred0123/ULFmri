"""Render a preprocessed [3,H,W] .npy (channels = T1w, T2w, FLAIR, float[0,1])
into a grayscale T1|T2|FLAIR montage PNG so it can actually be viewed.

Usage:
  python view_npy.py path/to/slice_129.npy               # -> slice_129.png next to it
  python view_npy.py path/to/slice_129.npy out.png
"""
import sys
import numpy as np
from PIL import Image

def render(npy_path, out_path=None):
    a = np.load(npy_path)                      # [3, H, W] float in [0,1]
    assert a.ndim == 3 and a.shape[0] == 3, f"expected [3,H,W], got {a.shape}"
    row = np.concatenate([np.clip(a[c], 0, 1) for c in range(3)], axis=1)  # T1|T2|FLAIR
    img = Image.fromarray((row * 255).astype("uint8"))   # grayscale, NOT RGB
    out_path = out_path or str(npy_path).rsplit(".", 1)[0] + ".png"
    img.save(out_path)
    print(f"saved {out_path}  (left→right: T1w | T2w | FLAIR)")

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
