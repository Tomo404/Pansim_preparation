from __future__ import annotations

import os
import pandas as pd

from data_loader import load_counts_from_tensor_files


TENSOR_PATH = "tensor_main.npy"
DIMS_PATH = "tensor_main_dimensions.json"
GENERATED_CSV = "generated_companies.csv"
OUTDIR = "analysis_out"
TOP_N = 10


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Input tensor counts -> DataFrame
    counts = load_counts_from_tensor_files(TENSOR_PATH, DIMS_PATH)

    inp = pd.DataFrame([
        {
            "settlement": k.settlement,
            "teaor": str(k.teaor),
            "bin": k.bin_name,
            "input_count": int(v),
        }
        for k, v in counts.items()
    ])

    inp["settlement"] = inp["settlement"].astype(str).str.strip()
    inp["teaor"] = inp["teaor"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    inp.loc[inp["teaor"] == "", "teaor"] = "0"
    inp["bin"] = inp["bin"].astype(str).str.strip()

    # 2) Generated -> aggregated DataFrame
    gen = pd.read_csv(
        GENERATED_CSV,
        dtype={"settlement": "string", "teaor": "string", "bin": "string"}
    )

    gen["settlement"] = gen["settlement"].astype(str).str.strip()
    gen["teaor"] = gen["teaor"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    gen.loc[gen["teaor"] == "", "teaor"] = "0"
    gen["bin"] = gen["bin"].astype(str).str.strip()

    gen_agg = (
        gen.groupby(["settlement", "teaor", "bin"])
        .size()
        .reset_index(name="generated_count")
    )

    # 3) Top N settlements by input total
    top_settlements = (
        inp.groupby("settlement")["input_count"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
        .index.tolist()
    )

    inp_top = inp[inp["settlement"].isin(top_settlements)].copy()
    gen_top = gen_agg[gen_agg["settlement"].isin(top_settlements)].copy()

    # 4) Merge
    comp = inp_top.merge(
        gen_top,
        on=["settlement", "teaor", "bin"],
        how="outer"
    ).fillna(0)

    comp["input_count"] = comp["input_count"].astype(int)
    comp["generated_count"] = comp["generated_count"].astype(int)
    comp["diff"] = comp["generated_count"] - comp["input_count"]

    # 5) Ratio
    comp["ratio"] = comp["generated_count"] / comp["input_count"].replace(0, pd.NA)

    # 6) Shares within settlement × teaor
    inp_tot = comp.groupby(["settlement", "teaor"])["input_count"].transform("sum")
    gen_tot = comp.groupby(["settlement", "teaor"])["generated_count"].transform("sum")

    comp["input_share_within_settlement_teaor"] = (
        comp["input_count"] / inp_tot.replace(0, pd.NA)
    )
    comp["generated_share_within_settlement_teaor"] = (
        comp["generated_count"] / gen_tot.replace(0, pd.NA)
    )
    comp["share_diff"] = (
        comp["generated_share_within_settlement_teaor"] -
        comp["input_share_within_settlement_teaor"]
    )

    # 7) Sort
    bin_order = ["1-4", "5-9", "10-19", "20-49", "50-249", "250-1000"]
    comp["bin"] = pd.Categorical(comp["bin"], categories=bin_order, ordered=True)
    comp = comp.sort_values(["settlement", "teaor", "bin"])

    # 8) Write full compare
    out_path = os.path.join(OUTDIR, "settlement_teaor_bin_compare_top10.csv")
    comp.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

    # 9) Also write the top 10 settlement list
    tops_path = os.path.join(OUTDIR, "top10_settlements_used.csv")
    pd.DataFrame({"settlement": top_settlements}).to_csv(tops_path, index=False)
    print(f"Wrote: {tops_path}")


if __name__ == "__main__":
    main()