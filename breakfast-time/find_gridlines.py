"""Find actual gridline positions by scanning row/column projections of dark pixels."""

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
img = cv2.imread(str(ROOT / "breakfast_time_grid.png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
dark = (gray < 60).astype(int)

row_dark = dark.sum(axis=1)
col_dark = dark.sum(axis=0)

# Find peaks: positions where row_dark > (mean + N*std).
def peaks(profile, min_height=None, min_dist=20):
    if min_height is None:
        # Default: 2x the median.
        min_height = 2 * np.median(profile)
    out = []
    i = 0
    n = len(profile)
    while i < n:
        if profile[i] >= min_height:
            # Local maximum within a window.
            j = i
            while j < n and profile[j] >= min_height:
                j += 1
            # Position = argmax in [i, j).
            pos = i + int(np.argmax(profile[i:j]))
            if not out or pos - out[-1] >= min_dist:
                out.append(pos)
            elif profile[pos] > profile[out[-1]]:
                out[-1] = pos
            i = j
        else:
            i += 1
    return out

h_lines = peaks(row_dark, min_height=200, min_dist=15)
v_lines = peaks(col_dark, min_height=200, min_dist=15)
print(f"Horizontal grid lines (count={len(h_lines)}): {h_lines}")
print(f"Vertical   grid lines (count={len(v_lines)}): {v_lines}")
print()
if len(h_lines) >= 2:
    print(f"H spans y={h_lines[0]}..{h_lines[-1]}, total {h_lines[-1]-h_lines[0]} px")
    print(f"H gaps: {np.diff(h_lines).tolist()}")
if len(v_lines) >= 2:
    print(f"V spans x={v_lines[0]}..{v_lines[-1]}, total {v_lines[-1]-v_lines[0]} px")
    print(f"V gaps: {np.diff(v_lines).tolist()}")
