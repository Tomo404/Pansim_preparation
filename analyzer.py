# analyzer.py
# ------------------------------
# Analyze generated_companies.csv:
# - bin-level stats (count, mean, median, spike ratios)
# - TEÁOR-level summary
# - optional detailed in-bin distributions saved to CSV
#
# Usage examples:
#   python analyzer.py --csv generated_companies.csv
#   python analyzer.py --csv generated_companies.csv --outdir analysis_out
#   python analyzer.py --csv generated_companies.csv --teaor 05 85 --write-bin-details
#
# Works best with pandas; falls back to csv module (limited) if pandas missing.

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

# Keep consistent with generator.py
BINS: Dict[str, Tuple[int, int]] = {
    "1-4": (1, 4),
    "5-9": (5, 9),
    "10-19": (10, 19),
    "20-49": (20, 49),
    "50-249": (50, 249),
    "250-1000": (250, 1000),
}


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def _format_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def analyze_with_pandas(
    csv_path: str,
    outdir: str,
    teaor_filter: Optional[List[str]] = None,
    write_bin_details: bool = False,
    chunksize: int = 500_000,
) -> None:
    """
    Chunked analysis for large files.
    """
    pd = _try_import_pandas()
    if pd is None:
        raise RuntimeError("pandas is required for full analysis. Install pandas or use a smaller custom parser.")

    # Accumulators
    # bin -> counts, sum, sumsq, minCount, maxCount, total
    bin_total: Dict[str, int] = {b: 0 for b in BINS}
    bin_sum: Dict[str, int] = {b: 0 for b in BINS}
    bin_min_hits: Dict[str, int] = {b: 0 for b in BINS}
    bin_max_hits: Dict[str, int] = {b: 0 for b in BINS}

    # teaor aggregations
    # teaor -> total rows, sum sizes
    teaor_total: Dict[str, int] = {}
    teaor_sum: Dict[str, int] = {}

    # TEÁOR x bin aggregations
    # (teaor, bin) -> total rows, sum sizes, min hits, max hits
    teaor_bin_total: Dict[tuple, int] = {}
    teaor_bin_sum: Dict[tuple, int] = {}
    teaor_bin_min_hits: Dict[tuple, int] = {}
    teaor_bin_max_hits: Dict[tuple, int] = {}

    # Optional: detailed in-bin distributions
    # bin -> size -> count
    bin_size_counts: Dict[str, Dict[int, int]] = {b: {} for b in BINS} if write_bin_details else {}

    # Read in chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Optional TEÁOR filter
        if teaor_filter:
            chunk = chunk[chunk["teaor"].astype(str).isin(set(teaor_filter))]

        # Ensure types
        chunk["bin"] = chunk["bin"].astype(str)
        chunk["teaor"] = chunk["teaor"].astype(str)
        chunk["company_size"] = chunk["company_size"].astype(int)

        # Bin-level
        for bname, (L, U) in BINS.items():
            sub = chunk[chunk["bin"] == bname]
            if sub.empty:
                continue

            n = int(len(sub))
            s = int(sub["company_size"].sum())
            bin_total[bname] += n
            bin_sum[bname] += s
            bin_min_hits[bname] += int((sub["company_size"] == L).sum())
            bin_max_hits[bname] += int((sub["company_size"] == U).sum())

            if write_bin_details:
                vc = sub["company_size"].value_counts()
                d = bin_size_counts[bname]
                for size_val, c in vc.items():
                    d[int(size_val)] = d.get(int(size_val), 0) + int(c)

        # TEÁOR-level (count + sum sizes)
        g = chunk.groupby("teaor")["company_size"].agg(["count", "sum"])
        for tea, row in g.iterrows():
            tea = str(tea)
            teaor_total[tea] = teaor_total.get(tea, 0) + int(row["count"])
            teaor_sum[tea] = teaor_sum.get(tea, 0) + int(row["sum"])

        # TEÁOR x BIN level stats
        # We'll compute count, sum, and min/max hits per (teaor, bin).
        g2 = chunk.groupby(["teaor", "bin"])["company_size"].agg(["count", "sum"])
        for (tea, bname), row in g2.iterrows():
            tea = str(tea)
            bname = str(bname)
            key = (tea, bname)

            teaor_bin_total[key] = teaor_bin_total.get(key, 0) + int(row["count"])
            teaor_bin_sum[key] = teaor_bin_sum.get(key, 0) + int(row["sum"])

        # min/max hits need bin bounds, so do it bin-wise (still fast)
        for bname, (L, U) in BINS.items():
            subb = chunk[chunk["bin"] == bname]
            if subb.empty:
                continue

            # group within this bin by teaor to count edge hits
            min_g = subb[subb["company_size"] == L].groupby("teaor")["company_size"].size()
            max_g = subb[subb["company_size"] == U].groupby("teaor")["company_size"].size()

            for tea, c in min_g.items():
                key = (str(tea), bname)
                teaor_bin_min_hits[key] = teaor_bin_min_hits.get(key, 0) + int(c)

            for tea, c in max_g.items():
                key = (str(tea), bname)
                teaor_bin_max_hits[key] = teaor_bin_max_hits.get(key, 0) + int(c)


    # ---- Print summary ----
    print("\n=== BIN SUMMARY ===")
    print("bin,count,mean_size,min_hit_ratio,max_hit_ratio")
    for bname, (L, U) in BINS.items():
        n = bin_total[bname]
        mean = _safe_div(bin_sum[bname], n)
        min_ratio = _safe_div(bin_min_hits[bname], n)
        max_ratio = _safe_div(bin_max_hits[bname], n)
        print(f"{bname},{n},{mean:.3f},{_format_pct(min_ratio)},{_format_pct(max_ratio)}")

    # Save bin summary CSV
    ensure_outdir(outdir)
    bin_summary_path = os.path.join(outdir, "bin_summary.csv")
    with open(bin_summary_path, "w", encoding="utf-8") as f:
        f.write("bin,count,mean_size,min_hit_count,min_hit_ratio,max_hit_count,max_hit_ratio\n")
        for bname, (L, U) in BINS.items():
            n = bin_total[bname]
            mean = _safe_div(bin_sum[bname], n)
            min_ratio = _safe_div(bin_min_hits[bname], n)
            max_ratio = _safe_div(bin_max_hits[bname], n)
            f.write(
                f"{bname},{n},{mean:.6f},{bin_min_hits[bname]},{min_ratio:.6f},{bin_max_hits[bname]},{max_ratio:.6f}\n"
            )
    print(f"\nWrote: {bin_summary_path}")

    # TEÁOR summary (top by count)
    tea_rows = [(t, teaor_total[t], _safe_div(teaor_sum[t], teaor_total[t])) for t in teaor_total]
    tea_rows.sort(key=lambda x: x[1], reverse=True)

    print("\n=== TOP TEÁOR BY COMPANY COUNT ===")
    print("teaor,count,mean_size")
    for t, cnt, mean in tea_rows[:30]:
        print(f"{t},{cnt},{mean:.3f}")

    tea_summary_path = os.path.join(outdir, "teaor_summary.csv")
    with open(tea_summary_path, "w", encoding="utf-8") as f:
        f.write("teaor,count,mean_size\n")
        for t, cnt, mean in tea_rows:
            f.write(f"{t},{cnt},{mean:.6f}\n")
    print(f"\nWrote: {tea_summary_path}")

    # TEÁOR x BIN summary
    teaor_bin_path = os.path.join(outdir, "teaor_bin_summary.csv")
    with open(teaor_bin_path, "w", encoding="utf-8") as f:
        f.write("teaor,bin,count,mean_size,min_hit_count,min_hit_ratio,max_hit_count,max_hit_ratio\n")

        # Sort by teaor then bin for easy Excel pivoting
        def bin_sort_key(bname: str) -> int:
            return int(bname.split("-")[0])

        keys_sorted = sorted(
            teaor_bin_total.keys(),
            key=lambda x: (x[0], bin_sort_key(x[1]))
        )

        for (tea, bname) in keys_sorted:
            n = teaor_bin_total.get((tea, bname), 0)
            if n == 0:
                continue

            s = teaor_bin_sum.get((tea, bname), 0)
            mean = _safe_div(s, n)

            min_hits = teaor_bin_min_hits.get((tea, bname), 0)
            max_hits = teaor_bin_max_hits.get((tea, bname), 0)

            min_ratio = _safe_div(min_hits, n)
            max_ratio = _safe_div(max_hits, n)

            f.write(f"{tea},{bname},{n},{mean:.6f},{min_hits},{min_ratio:.6f},{max_hits},{max_ratio:.6f}\n")

    print(f"Wrote: {teaor_bin_path}")

    # Optional: write in-bin detailed distributions
    if write_bin_details:
        for bname, (L, U) in BINS.items():
            d = bin_size_counts.get(bname, {})
            if not d:
                continue

            detail_path = os.path.join(outdir, f"bin_detail_{bname}.csv")
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write("company_size,count,ratio\n")
                total = sum(d.values())
                for size_val in range(L, U + 1):
                    c = d.get(size_val, 0)
                    r = _safe_div(c, total)
                    f.write(f"{size_val},{c},{r:.8f}\n")
            print(f"Wrote: {detail_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to generated_companies.csv")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory for summaries")
    ap.add_argument("--teaor", nargs="*", default=None, help="Optional TEÁOR codes to filter (e.g. 05 85)")
    ap.add_argument("--write-bin-details", action="store_true", help="Write per-bin internal distributions")
    ap.add_argument("--chunksize", type=int, default=500_000, help="CSV read chunk size (pandas)")
    args = ap.parse_args()

    analyze_with_pandas(
        csv_path=args.csv,
        outdir=args.outdir,
        teaor_filter=args.teaor,
        write_bin_details=args.write_bin_details,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
