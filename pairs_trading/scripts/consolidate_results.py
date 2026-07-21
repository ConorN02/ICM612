"""Consolidate every CSV in pairs_trading/results/ into a single Excel workbook.

Reads each `*.csv` file in `config.RESULTS_DIR` and writes it to its own
sheet in `results/all_results.xlsx` (sheet name = filename without the
".csv" extension, truncated to Excel's 31-character sheet name limit, with
a numeric suffix appended if truncation causes a collision). Handy for
dropping the whole pipeline's output into the report as a single file
rather than attaching 17+ separate CSVs.

Usage:
    python3 pairs_trading/scripts/consolidate_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly (`python3 pairs_trading/scripts/...py`)
# regardless of the current working directory, by putting the repo root
# (parent of the pairs_trading package) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pairs_trading import config

MAX_SHEET_NAME_LENGTH = 31


def _make_sheet_name(csv_path: Path, used_names: set[str]) -> str:
    """Derive a valid, unique Excel sheet name from a CSV file's name.

    Excel sheet names are capped at `MAX_SHEET_NAME_LENGTH` characters and
    must be unique within a workbook. Long filenames are truncated; if
    that truncation collides with a name already used in this workbook,
    a short numeric suffix is appended (still within the length limit)
    until a unique name is found.

    Args:
        csv_path: Path to the source CSV file.
        used_names: Sheet names already assigned earlier in this workbook.
            Not mutated here -- the caller must add the returned name to
            `used_names` before processing the next file.

    Returns:
        A sheet name of at most `MAX_SHEET_NAME_LENGTH` characters, not
        already present in `used_names`.

    Raises:
        ValueError: If no unique name can be found (exhausted suffixes).
    """
    base_name = csv_path.stem[:MAX_SHEET_NAME_LENGTH]
    if base_name not in used_names:
        return base_name

    for suffix in range(2, 100):
        suffix_str = f"_{suffix}"
        candidate = base_name[: MAX_SHEET_NAME_LENGTH - len(suffix_str)] + suffix_str
        if candidate not in used_names:
            return candidate

    raise ValueError(f"Could not derive a unique Excel sheet name for {csv_path.name}")


def main() -> None:
    """Read every CSV in config.RESULTS_DIR and write them into one Excel workbook."""
    csv_paths = sorted(config.RESULTS_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"[consolidate_results] No CSV files found in {config.RESULTS_DIR}; nothing to do.")
        return

    out_path = config.RESULTS_DIR / "all_results.xlsx"
    used_names: set[str] = set()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            sheet_name = _make_sheet_name(csv_path, used_names)
            used_names.add(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[consolidate_results] {csv_path.name} -> sheet '{sheet_name}' ({len(df)} rows)")

    print(f"\n[consolidate_results] Wrote {len(csv_paths)} sheet(s) to {out_path}")


if __name__ == "__main__":
    main()
