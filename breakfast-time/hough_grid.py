"""
Find the puzzle's gridline positions via Hough transform.

Steps:
  1. Threshold image to isolate dark pixels (gridlines + black cells).
  2. Use cv2.HoughLinesP on the dark-edge image to find long straight lines.
  3. Cluster horizontal lines by Y and vertical lines by X.
  4. The 22 strongest horizontal lines and 21 strongest vertical lines define
     cell boundaries.
  5. Sample each cell.
"""

from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ROWS, COLS = 21, 20

bgr = cv2.imread(str(ROOT / "breakfast_time_grid.png"))
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# Edge detection on the original grayscale.
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite(str(ROOT / "edges.png"), edges)

# Hough probabilistic — find segments longer than ~half the image dimension.
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                        minLineLength=min(h, w) // 2, maxLineGap=20)
print(f"Hough lines found: {0 if lines is None else len(lines)}")

if lines is None:
    raise SystemExit("No lines found.")

# Separate horizontal and vertical lines.
h_ys = []   # y positions of horizontal lines
v_xs = []   # x positions of vertical lines
for ln in lines:
    x1, y1, x2, y2 = ln[0]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx > 5 * dy:  # near-horizontal
        h_ys.append((y1 + y2) / 2)
    elif dy > 5 * dx:  # near-vertical
        v_xs.append((x1 + x2) / 2)

def cluster(values, gap):
    """Cluster sorted values, returning the mean of each cluster."""
    if not values: return []
    values = sorted(values)
    out = [[values[0]]]
    for v in values[1:]:
        if v - out[-1][-1] < gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]

h_ys_clustered = cluster(h_ys, gap=10)
v_xs_clustered = cluster(v_xs, gap=10)
print(f"\nClustered horizontal lines ({len(h_ys_clustered)}): {[round(y, 1) for y in h_ys_clustered]}")
print(f"Clustered vertical   lines ({len(v_xs_clustered)}): {[round(x, 1) for x in v_xs_clustered]}")
