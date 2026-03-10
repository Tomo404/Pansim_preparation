# Pansim_preparation
Workplace generation codes for Pansim

# Pipeline:

1. main.py
   - loads tensor
   - optional scaling
   - generates companies

2. analyzer.py
   - TEÁOR × bin statistics

3. validator.py
   - settlement and company count checks

4. stitched_hist.py
   - TEÁOR-level company size distribution

5. stitched_settlement.py
   - settlement-level size distribution

6. settlement_teaor_bin_compare.py
   - settlement × TEÁOR × bin validation
# STEPS

1. Run main.py
2. Inside terminal, use: python analyzer.py --csv generated_companies.csv --outdir analysis_out
3. Then still in terminal, run: python validator.py --generated generated_companies.csv --input-counts tensor_counts_whitelist.csv
4. Finally, stitch: python stitched_hist.py --csv generated_companies.csv --top 40 --out analysis_out
