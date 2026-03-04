# validator.py
# ---------------------------------
# Validation checks:
# 1) TEÁOR workers: compare KSH workers_by_teaor vs generated sum(company_size)
# 2) Territorial/company-count consistency: compare input counts vs generated counts
#
# Usage examples:
#   python validator.py --generated generated_companies.csv --ksh-workers ksh_workers_by_teaor.csv
#   python validator.py --generated generated_companies.csv --input-counts ksh_company_counts.csv

from __future__ import annotations

import argparse
import os
from typing import Optional

def try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None

def read_table(pd, path: str):
    """
    Read either CSV or Excel based on file extension.
    Excel: first sheet by default.
    """
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")

import re

VALID_BINS = {"1-4","5-9","10-19","20-49","50-249","250-1000"}
BIN_ORDER = ["1-4","5-9","10-19","20-49","50-249","250-1000"]

def normalize_bin(x):
    """
    Normalize any KSH / HU bin text into one of:
    1-4, 5-9, 10-19, 20-49, 50-249, 250-1000
    """
    if x is None or (isinstance(x, float) and str(x) == "nan"):
        return None

    s = str(x).strip().lower()

    # Fix common encoding oddities (ő->õ etc.)
    # and remove whitespace/quotes
    s = s.replace("ő", "õ").replace("ű", "û")
    s = s.replace('"', "").replace("'", "").replace(" ", "")

    # Anything that looks like "250+ / 250felett / 250fõfelett / 250fõfeletti" => 250-1000
    if "250" in s and ("felett" in s or "fele" in s or "+" in s):
        return "250-1000"

    # If it already contains a range like 1-4 / 5-9 / 10-19 / 20-49 / 50-249
    m = re.search(r"(\d+)\-(\d+)", s)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        cand = f"{a}-{b}"
        return cand if cand in VALID_BINS else None

    # If it contains standalone "250" (sometimes it gets truncated)
    if "250" in s:
        return "250-1000"

    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default=None, help="Path to tensor_main.npy (optional)")
    ap.add_argument("--tensor-dims", default=None, help="Path to tensor_main_dimensions.json (optional)")
    ap.add_argument("--generated", required=True, help="Path to generated_companies.csv")
    ap.add_argument("--ksh-workers", default=None, help="CSV with columns: teaor,workers")
    ap.add_argument("--input-counts", default=None, help="CSV with columns: settlement,teaor,bin,count (KSH company counts)")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory")
    args = ap.parse_args()

    pd = try_import_pandas()
    if pd is None:
        raise RuntimeError("pandas required: pip install pandas")

    os.makedirs(args.outdir, exist_ok=True)

    gen = pd.read_csv(args.generated)
    gen["teaor"] = gen["teaor"].astype(str)
    gen["teaor"] = gen["teaor"].str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    gen.loc[gen["teaor"] == "", "teaor"] = "0"
    gen["settlement"] = gen["settlement"].astype(str)
    gen["bin"] = gen["bin"].astype(str)
    gen["company_size"] = gen["company_size"].astype(int)

    # -----------------------------
    # (1) TEÁOR workers compare
    # -----------------------------
    if args.ksh_workers:
        kshw = read_table(pd, args.ksh_workers)
        kshw["teaor"] = kshw["teaor"].astype(str)
        kshw["workers"] = kshw["workers"].astype(float)

        gen_workers = gen.groupby("teaor")["company_size"].sum().reset_index()
        gen_workers = gen_workers.rename(columns={"company_size": "generated_workers"})

        comp = kshw.merge(gen_workers, on="teaor", how="left").fillna({"generated_workers": 0})
        comp["abs_diff"] = comp["generated_workers"] - comp["workers"]
        comp["rel_diff"] = comp["abs_diff"] / comp["workers"].replace(0, float("nan"))

        comp = comp.sort_values("workers", ascending=False)

        out_path = os.path.join(args.outdir, "teaor_workers_compare.csv")
        comp.to_csv(out_path, index=False)

        print(f"Wrote: {out_path}")
        print("Top 10 by KSH workers:")
        print(comp[["teaor", "workers", "generated_workers", "abs_diff", "rel_diff"]].head(10).to_string(index=False))

    # -----------------------------
    # (2) Territorial/company-count consistency
    # -----------------------------
    from data_loader import load_counts_from_tensor_files
    if args.input_counts:
        if args.tensor and args.tensor_dims:
            counts = load_counts_from_tensor_files(args.tensor, args.tensor_dims)
            # counts: Dict[CellKey,int] -> DataFrame
            inp = pd.DataFrame([
                {"settlement": k.settlement, "teaor": k.teaor, "bin": k.bin_name, "count": int(v)}
                for k, v in counts.items()
            ])
        else:
            inp = read_table(pd, args.input_counts)
            # ... maradhat a meglévő rename_map / normalize_bin logika
        # ---- rename Hungarian columns -> expected names ----
        rename_map = {
            "Terület": "settlement",
            "TEÁOR alág kód": "teaor",
            "Érték": "count",
            "Létszámkategória": "bin",
        }
        inp = inp.rename(columns=rename_map)

        # Required columns check
        required = ["settlement", "teaor", "bin", "count"]
        missing = [c for c in required if c not in inp.columns]
        if missing:
            raise RuntimeError(
                f"input-counts is missing required columns after rename: {missing}. "
                f"Available columns: {list(inp.columns)}"
            )

        inp["settlement"] = inp["settlement"].astype(str)
        inp["teaor"] = inp["teaor"].astype(str)
        # Fix Excel float teaor codes like "1.0" -> "1"
        inp["teaor"] = inp["teaor"].str.replace(r"\.0$", "", regex=True).str.lstrip("0")
        inp.loc[inp["teaor"] == "", "teaor"] = "0"
        inp["count"] = inp["count"].astype(int)

        # ---- normalize bins robustly ----
        inp["bin"] = inp["bin"].apply(normalize_bin)

        # Drop rows where bin could not be parsed (these are layout / empty rows)
        before = len(inp)
        inp = inp[~inp["bin"].isna()].copy()
        after = len(inp)
        print(f"Dropped layout/invalid bin rows: {before - after}")

        # Drop rows where we still couldn't parse a bin (these are not comparable cells)
        bad = inp[inp["bin"].isna()].copy()
        if len(bad) > 0:
            bad_path = os.path.join(args.outdir, "input_rows_with_missing_bin.csv")
            bad.to_csv(bad_path, index=False)
            print(f"WARNING: {len(bad)} input rows had missing/unparsed bin. Wrote: {bad_path}")
            inp = inp[~inp["bin"].isna()].copy()

        # Only keep valid bins
        inp = inp[inp["bin"].isin(VALID_BINS)].copy()
        print("Input bin value counts:")
        print(inp["bin"].value_counts().to_string())

        # generated company counts per cell
        gen_cell = gen.groupby(["settlement", "teaor", "bin"]).size().reset_index(name="generated_count")

        comp2 = inp.merge(gen_cell, on=["settlement", "teaor", "bin"], how="left").fillna({"generated_count": 0})
        comp2["generated_count"] = comp2["generated_count"].astype(int)
        comp2["diff"] = comp2["generated_count"] - comp2["count"]

        # mismatches
        mism = comp2[comp2["diff"] != 0].copy()
        mism = mism.sort_values(["settlement", "teaor", "bin"])

        out_all = os.path.join(args.outdir, "cell_count_compare.csv")
        out_mis = os.path.join(args.outdir, "cell_count_mismatches.csv")
        comp2.to_csv(out_all, index=False)
        mism.to_csv(out_mis, index=False)

        print(f"Wrote: {out_all}")
        print(f"Wrote: {out_mis}")
        print(f"Mismatching cells: {len(mism)} (should be 0 if generation preserves counts)")

        # -----------------------------
        # (2b) TEÁOR-bin national compare (stitch bins together by TEÁOR)
        # This is what the supervisor asked for: distribution shape by TEÁOR.
        # -----------------------------

        # KSH: sum counts over settlements -> (teaor, bin)
        ksh_tb = inp.groupby(["teaor", "bin"])["count"].sum().reset_index()
        ksh_tb = ksh_tb.rename(columns={"count": "ksh_count"})

        # Generated: count companies over settlements -> (teaor, bin)
        gen_tb = gen.groupby(["teaor", "bin"]).size().reset_index(name="generated_count")

        teaor_bin = ksh_tb.merge(gen_tb, on=["teaor", "bin"], how="outer").fillna(0)
        teaor_bin["ksh_count"] = teaor_bin["ksh_count"].astype(int)
        teaor_bin["generated_count"] = teaor_bin["generated_count"].astype(int)
        teaor_bin["diff"] = teaor_bin["generated_count"] - teaor_bin["ksh_count"]

        # Ratios within TEÁOR (shape comparison)
        teaor_tot_ksh = teaor_bin.groupby("teaor")["ksh_count"].transform("sum")
        teaor_tot_gen = teaor_bin.groupby("teaor")["generated_count"].transform("sum")

        # Avoid div by zero
        teaor_bin["ksh_ratio"] = teaor_bin["ksh_count"] / teaor_tot_ksh.replace(0, float("nan"))
        teaor_bin["generated_ratio"] = teaor_bin["generated_count"] / teaor_tot_gen.replace(0, float("nan"))
        teaor_bin["ratio_diff"] = teaor_bin["generated_ratio"] - teaor_bin["ksh_ratio"]

        # Pretty ordering of bins
        teaor_bin["bin"] = pd.Categorical(teaor_bin["bin"], categories=BIN_ORDER, ordered=True)
        teaor_bin = teaor_bin.sort_values(["teaor", "bin"])

        out_tb = os.path.join(args.outdir, "teaor_bin_compare.csv")
        teaor_bin.to_csv(out_tb, index=False)
        print(f"Wrote: {out_tb}")

        # Also write a "wide" pivot (easy Excel chart)
        wide_counts = teaor_bin.pivot_table(index="teaor", columns="bin", values="generated_count", fill_value=0)
        wide_counts_path = os.path.join(args.outdir, "teaor_bin_generated_wide.csv")
        wide_counts.to_csv(wide_counts_path)
        print(f"Wrote: {wide_counts_path}")

        wide_ratios = teaor_bin.pivot_table(index="teaor", columns="bin", values="generated_ratio", fill_value=0)
        wide_ratios_path = os.path.join(args.outdir, "teaor_bin_generated_ratio_wide.csv")
        wide_ratios.to_csv(wide_ratios_path)
        print(f"Wrote: {wide_ratios_path}")

        # also compare settlement totals
        inp_sett = inp.groupby("settlement")["count"].sum().reset_index(name="ksh_companies")
        gen_sett = gen.groupby("settlement").size().reset_index(name="generated_companies")
        sett_comp = inp_sett.merge(gen_sett, on="settlement", how="left").fillna({"generated_companies": 0})
        sett_comp["diff"] = sett_comp["generated_companies"].astype(int) - sett_comp["ksh_companies"].astype(int)

        out_sett = os.path.join(args.outdir, "settlement_company_totals_compare.csv")
        sett_comp.to_csv(out_sett, index=False)
        print(f"Wrote: {out_sett}")

if __name__ == "__main__":
    main()