"""
Detect the black/white cell pattern of breakfast_time_grid.png by finding
white-cell blobs via connected components, then mapping each blob to a grid
cell with PER-ROW X-alignment to absorb perspective distortion.
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG  = ROOT / "breakfast_time_grid.png"
DATA = ROOT / "puzzle-data.json"

ROWS, COLS = 21, 20


def main():
    bgr = cv2.imread(str(IMG))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bright = (gray > 180).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bright_e = cv2.erode(bright, kernel, iterations=1)

    num, _, stats, centroids = cv2.connectedComponentsWithStats(bright_e, connectivity=4)
    blobs = []
    for i in range(1, num):
        area = stats[i, 4]
        if 500 <= area <= 2500:
            blobs.append((float(centroids[i][0]), float(centroids[i][1]), int(area)))
    print(f"White-cell blobs: {len(blobs)}")

    # Step 1: assign each blob to a row using its Y coordinate.
    # Find global Y origin and stride first by clustering Y centers.
    ys = sorted(b[1] for b in blobs)
    # The y values of blobs in the same row are within ~5 px of each other.
    y_clusters = [[ys[0]]]
    for y in ys[1:]:
        if y - y_clusters[-1][-1] < 18:
            y_clusters[-1].append(y)
        else:
            y_clusters.append([y])
    y_means = [float(np.mean(c)) for c in y_clusters]
    print(f"Distinct Y rows found: {len(y_means)}")
    if len(y_means) > ROWS:
        # Sometimes one row gets split — merge nearest.
        while len(y_means) > ROWS:
            diffs = np.diff(y_means)
            idx = int(np.argmin(diffs))
            merged = (y_means[idx] + y_means[idx + 1]) / 2
            y_means = y_means[:idx] + [merged] + y_means[idx + 2:]

    # Pad to ROWS if some rows were entirely black (no blobs in them).
    if len(y_means) < ROWS:
        # Estimate stride and pad missing rows.
        diffs = np.diff(y_means)
        med_stride = float(np.median(diffs))
        print(f"  median Y stride: {med_stride:.2f}")
        full = []
        prev = y_means[0]
        full.append(prev)
        for y in y_means[1:]:
            gap = y - prev
            n_skip = max(0, round(gap / med_stride) - 1)
            for k in range(1, n_skip + 1):
                full.append(prev + k * (gap / (n_skip + 1)))
            full.append(y)
            prev = y
        # Maybe still short; pad to ROWS by extending below.
        while len(full) < ROWS:
            full.append(full[-1] + med_stride)
        y_means = full[:ROWS]
        print(f"  padded to {len(y_means)} rows")

    print(f"Y row centers: {[round(y,1) for y in y_means]}")

    # Step 2: assign each blob to a row by nearest Y.
    blobs_by_row = [[] for _ in range(ROWS)]
    for b in blobs:
        bx, by, _ = b
        r = min(range(ROWS), key=lambda i: abs(by - y_means[i]))
        blobs_by_row[r].append(b)

    # Step 3: for each row, build the column assignment.
    # All blobs in this row should land on integer cell columns.
    # Find optimal x_origin & col_stride for the WHOLE grid first (global), then
    # let each row's mapping shift slightly.
    all_xs = sorted(b[0] for b in blobs)
    xmin = all_xs[0]
    xmax = all_xs[-1]
    # The leftmost blob is in some col K (probably 0). The rightmost is in col L (probably 19).
    # Estimate stride assuming xmin = col 0 and xmax = col 19.
    col_stride_global = (xmax - xmin) / (COLS - 1)
    print(f"Global col_stride estimate: {col_stride_global:.2f}")

    cells = [["#" for _ in range(COLS)] for _ in range(ROWS)]
    for r in range(ROWS):
        row_blobs = blobs_by_row[r]
        if not row_blobs:
            continue
        row_xs = sorted(b[0] for b in row_blobs)
        # Per-row alignment: shift x_origin so blobs align best with integer columns.
        # Try a few candidate origins within ±cell_stride/2 of the global one.
        # Global origin = xmin.
        best_err = 1e18
        best_org = xmin
        for dx in np.arange(-col_stride_global * 0.5,
                            col_stride_global * 0.5 + 0.1, 0.5):
            org = xmin + dx
            err = 0.0
            for x in row_xs:
                col_f = (x - org) / col_stride_global
                col_i = round(col_f)
                if 0 <= col_i < COLS:
                    err += (col_f - col_i) ** 2
                else:
                    err += 4.0
            if err < best_err:
                best_err = err
                best_org = org
        for x in row_xs:
            col_f = (x - best_org) / col_stride_global
            col_i = round(col_f)
            if 0 <= col_i < COLS:
                cells[r][col_i] = "."

    print()
    print("Detected grid:")
    print("       " + "".join(f"{c%10}" for c in range(COLS)))
    for r in range(ROWS):
        print(f"  {r:2d}   " + "".join(cells[r]))

    # Vs JSON.
    data = json.loads(DATA.read_text())
    raw = data["grid"]["cells"]
    miss_bk, miss_op = [], []
    for r in range(ROWS):
        for c in range(COLS):
            v = raw[r][c]
            if v not in ("#", ".") and cells[r][c] == "#":
                miss_bk.append((r, c, v))
            if v == "#" and cells[r][c] == ".":
                miss_op.append((r, c))
    n_black = sum(1 for r in range(ROWS) for c in range(COLS) if cells[r][c] == "#")
    n_white = ROWS * COLS - n_black
    print(f"\nBlack cells: {n_black}, white cells: {n_white}")
    print(f"Numbered cells detected as BLACK: {len(miss_bk)}")
    for r, c, v in miss_bk[:30]:
        print(f"  ({r:2d},{c:2d}) numbered {v}")
    print(f"\nJSON-# cells detected as OPEN: {len(miss_op)}")
    for r, c in miss_op[:30]:
        print(f"  ({r:2d},{c:2d})")

    # Overlay.
    overlay = bgr.copy()
    for x, y, _ in blobs:
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), 2)
    cv2.imwrite(str(ROOT / "grid_overlay.png"), overlay)

    (ROOT / "grid_detected.json").write_text(json.dumps({
        "rows": ROWS, "cols": COLS,
        "y_centers": y_means,
        "col_stride": col_stride_global,
        "cells": cells,
    }, indent=2))


if __name__ == "__main__":
    main()
