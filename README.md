# Pansim_preparation
Workplace generation codes for Pansim

# STEPS

1. Run main.py
2. Inside terminal, use: python analyzer.py --csv generated_companies.csv --outdir analysis_out
3. Then still in terminal, run: python validator.py --generated generated_companies.csv --input-counts tensor_counts_whitelist.csv
4. Finally, stitch: python stitched_hist.py --csv generated_companies.csv --top 40 --out analysis_out
