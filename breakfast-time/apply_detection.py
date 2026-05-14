"""
Apply the detected black/white pattern to puzzle-data.json.

  - Start from the existing JSON cells (preserves numbers).
  - For each cell, if detection says "#" and JSON says "." (or vice versa),
    update — UNLESS the JSON cell has a number, in which case keep the number
    and warn (a numbered cell can't be black, so this signals a detection
    error or a JSON numbering error we'll need to look at).
  - After patching, re-derive word lengths and compare to the bank.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "puzzle-data.json"
DETECTED = ROOT / "grid_detected.json"

ROWS, COLS = 21, 20

def main():
    data = json.loads(DATA.read_text())
    detected = json.loads(DETECTED.read_text())
    raw = data["grid"]["cells"]
    det = detected["cells"]

    new_cells = []
    conflicts = []
    changed = 0
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            orig = raw[r][c]
            d = det[r][c]
            if orig not in ("#", "."):
                # Numbered cell — must stay open. Warn if detection disagrees.
                if d == "#":
                    conflicts.append((r, c, orig))
                row.append(orig)
            else:
                if d != orig:
                    changed += 1
                row.append(d)
        new_cells.append(row)

    print(f"Cells changed: {changed}")
    print(f"Numbered-cell conflicts (detection says black on a numbered cell): {len(conflicts)}")
    for r, c, n in conflicts:
        print(f"  ({r:2d},{c:2d}) numbered {n}")

    # Derive word lengths.
    def is_open(r, c):
        if r < 0 or r >= ROWS or c < 0 or c >= COLS:
            return False
        return new_cells[r][c] != "#"

    lengths = {}
    for r in range(ROWS):
        for c in range(COLS):
            if not is_open(r, c): continue
            if not is_open(r, c - 1):
                ln = 0
                while is_open(r, c + ln): ln += 1
                if ln >= 2:
                    lengths[ln] = lengths.get(ln, 0) + 1
            if not is_open(r - 1, c):
                ln = 0
                while is_open(r + ln, c): ln += 1
                if ln >= 2:
                    lengths[ln] = lengths.get(ln, 0) + 1

    bank = data["wordBank"]
    targets = {int(k): len(v) for k, v in bank.items()}
    print("\nWord-length distribution:")
    print("  len  detected  bank  diff")
    for ln in sorted(set(lengths) | set(targets)):
        d = lengths.get(ln, 0)
        t = targets.get(ln, 0)
        flag = "" if d == t else "  <-- mismatch"
        print(f"  {ln:3d}  {d:8d}  {t:4d}  {d-t:+5d}{flag}")
    print(f"  total detected: {sum(lengths.values())} / bank target: {sum(targets.values())}")

    # Print the new grid as ASCII for eyeball.
    print("\nNew grid:")
    print("       " + "".join(f"{c%10}" for c in range(COLS)))
    for r in range(ROWS):
        s = f"  {r:2d}   "
        for c in range(COLS):
            v = new_cells[r][c]
            if v == "#": s += "#"
            elif v == ".": s += "."
            else: s += "."
        print(s)

    # Write back to puzzle-data.json (preserve original metadata, just rewrite cells).
    data["grid"]["cells"] = new_cells
    out = ROOT / "puzzle-data.patched.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {out.name}")

if __name__ == "__main__":
    main()
