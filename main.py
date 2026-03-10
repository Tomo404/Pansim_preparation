# main.py
# ------------------------------
# End-to-end runner:
# 1) load counts from tensor_main.npy + tensor_main_dimensions.json
# 2) generate concrete company sizes per (settlement, teaor, bin)
# 3) export a flat CSV with one row per generated company

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional
from teaor_profiles import load_teaor_bin_profiles

import csv

from data_loader import load_counts_from_tensor_files
from generator import CellKey, generate_companies_from_counts


# =========================
# CONFIG (EDIT THIS BLOCK)
# =========================
TENSOR_PATH = "tensor_main.npy"
DIMS_PATH = "tensor_main_dimensions.json"
TARGET_WORKERS = 4_600_000
BASE_WORKERS = 3_700_000
ENABLE_COUNT_SCALING = True
OUTPUT_CSV = "generated_companies.csv"
TEAOR_PROFILE_XLSX = "teaor_bin_summary.xlsx"
TEAOR_PROFILE_SHEET = None
TEAOR_PROFILE_START_ROW = 1

SEED = 12345
MAX_CELLS = None  # első körben legyen kicsi (pl. 2000)

# Default sampling profile for bin-internal sizes:
DEFAULT_PROFILE = "decay_mild"   # try: uniform / decay_mild / decay_strong / exp_mild / exp_strong

# Optional: TEÁOR-specific profile overrides.
# Example:
# PROFILE_BY_TEAOR = {"05": "decay_mild", "85": "decay_strong"}
PROFILE_BY_TEAOR: Dict[str, str] = {}
# =========================
# Only generate these TEÁOR codes first (debug / pilot)
# TEAOR_WHITELIST = {"11","16","23","41","55","66","78","80","84","90"}
TEAOR_WHITELIST = None

def flatten_generated(
    generated: Dict[CellKey, List[int]]
) -> List[dict]:
    """
    Convert Dict[CellKey, List[int]] into a row list.
    One row = one generated company.
    """
    rows: List[dict] = []

    for key, sizes in generated.items():
        for idx, size in enumerate(sizes):
            rows.append(
                {
                    "settlement": key.settlement,
                    "teaor": key.teaor,
                    "bin": key.bin_name,
                    "company_index_in_cell": idx,
                    "company_size": int(size),
                }
            )

    return rows

import math

def scale_counts_proportionally(
    counts: Dict[CellKey, int],
    scale: float,
) -> Dict[CellKey, int]:
    if scale <= 0:
        raise ValueError("scale must be > 0")

    items = list(counts.items())

    scaled_floor = {}
    remainders = []

    for key, value in items:
        expected = value * scale
        base = int(math.floor(expected))
        scaled_floor[key] = base
        remainders.append((key, expected - base))

    target_total = int(round(sum(counts.values()) * scale))
    current_total = sum(scaled_floor.values())
    missing = target_total - current_total

    # legnagyobb tört részek kapnak +1-et
    remainders.sort(key=lambda x: x[1], reverse=True)

    for i in range(missing):
        key = remainders[i][0]
        scaled_floor[key] += 1

    return scaled_floor

def write_csv(rows: List[dict], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    # 1) Load KSH counts
    counts = load_counts_from_tensor_files(
        tensor_npy_path=TENSOR_PATH,
        dimensions_json_path=DIMS_PATH,
        drop_zeros=True,
    )

    if ENABLE_COUNT_SCALING:
        scale = TARGET_WORKERS / BASE_WORKERS
        original_total = sum(counts.values())
        counts = scale_counts_proportionally(counts, scale)
        scaled_total = sum(counts.values())

        print(f"Scaling enabled: factor = {scale:.6f}")
        print(f"Original organization count total: {original_total:,}")
        print(f"Scaled organization count total:   {scaled_total:,}")

    teaor_bin_probs = load_teaor_bin_profiles(
        TEAOR_PROFILE_XLSX,
        sheet_name=TEAOR_PROFILE_SHEET,
        start_row=TEAOR_PROFILE_START_ROW,
    )

    # TEÁOR -> profile selection based on small-company dominance
    # (you can tune thresholds later)
    profile_by_teaor = {}
    for tea, bp in teaor_bin_probs.items():
        small = bp.get("1-4", 0.0) + bp.get("5-9", 0.0)
        mid = bp.get("10-19", 0.0) + bp.get("20-49", 0.0)
        large = bp.get("50-249", 0.0) + bp.get("250-1000", 0.0)

        if large >= 0.25:
            profile_by_teaor[tea] = "uniform"       # laposabb binen belül
        elif small >= 0.75:
            profile_by_teaor[tea] = "decay_strong"  # meredekebb
        else:
            profile_by_teaor[tea] = "decay_mild"    # alap

    # ---- BIN-LEVEL edge smoothing (B1) ----
    # If adjacent bin mass is tiny, don't force smoothing (teacher: OK if bin is absent)
    SUPPORT_MIN = 0.01  # 1% of companies in that bin in this TEÁOR => "present"
    # thresholds on upper/lower bin proportion ratio
    R_STRONG = 0.15
    R_ULTRA  = 0.05

    # Bin order for checking boundaries
    BIN_ORDER = ["1-4","5-9","10-19","20-49","50-249","250-1000"]

    profile_by_teaor_bin = {}
    window_by_boundary = {}

    for tea, bp in teaor_bin_probs.items():
        base = profile_by_teaor.get(tea, DEFAULT_PROFILE)
        for b in BIN_ORDER:
            profile_by_teaor_bin[(tea, b)] = base

        for i in range(len(BIN_ORDER) - 1):
            b_lo = BIN_ORDER[i]
            b_hi = BIN_ORDER[i + 1]

            p_lo = bp.get(b_lo, 0.0)
            p_hi = bp.get(b_hi, 0.0)

            if p_hi < SUPPORT_MIN or p_lo <= 0:
                continue

            r = p_hi / p_lo

            if r < R_ULTRA:
                profile_by_teaor_bin[(tea, b_lo)] = "decay_ultra"
                profile_by_teaor_bin[(tea, b_hi)] = "exp_ultra"
                window_by_boundary[(tea, b_lo, b_hi)] = 1

            elif r < R_STRONG:
                profile_by_teaor_bin[(tea, b_lo)] = "decay_strong"
                profile_by_teaor_bin[(tea, b_hi)] = "exp_strong"
                window_by_boundary[(tea, b_lo, b_hi)] = 1

            else:
                profile_by_teaor_bin[(tea, b_lo)] = "decay_mild"
                profile_by_teaor_bin[(tea, b_hi)] = "exp_mild"
                window_by_boundary[(tea, b_lo, b_hi)] = 2

    # Filter to whitelist TEÁORs (pilot)
    if TEAOR_WHITELIST:
        counts = {k: v for k, v in counts.items() if k.teaor in TEAOR_WHITELIST}
        teaor_bin_probs = {t: bp for t, bp in teaor_bin_probs.items() if t in TEAOR_WHITELIST}
    # --- DEBUG: only keep first MAX_CELLS cells to test quickly ---
    if MAX_CELLS is not None:
        counts_items = list(counts.items())[:MAX_CELLS]
        counts = dict(counts_items)
        print(f"DEBUG: limiting to {len(counts):,} cells")
    # -------------------------------------------------------------

    print(f"Loaded non-zero cells: {len(counts):,}")

    # 2) Generate company sizes
    generated = generate_companies_from_counts(
        counts=counts,
        profile_by_teaor=profile_by_teaor,
        profile_by_teaor_bin=profile_by_teaor_bin,
        default_profile=DEFAULT_PROFILE,
        seed=SEED,
        window_by_boundary=window_by_boundary,
    )

    # quick sanity
    total_companies = sum(len(v) for v in generated.values())
    print(f"Generated companies: {total_companies:,}")

    # 3) Export
    rows = flatten_generated(generated)
    write_csv(rows, OUTPUT_CSV)
    print(f"Wrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
