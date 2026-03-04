from teaor_profiles import load_teaor_bin_profiles, pretty_print_profile

EXCEL_PATH = "teaor_bin_summary.xlsx"  # ha máshogy hívod, írd át
SHEET_NAME = None                      # ha tudod a sheet nevét, beírhatod
START_ROW = 1                          # ha van fejléc és az útban van, emeld (pl. 2 vagy 3)

profiles = load_teaor_bin_profiles(EXCEL_PATH, sheet_name=SHEET_NAME, start_row=START_ROW)
print(f"Loaded TEÁOR profiles: {len(profiles)}")

# Pár példa:
for tea in ["1", "2", "3", "43", "85"]:
    pretty_print_profile(profiles, tea)

"""import pandas as pd

df = pd.read_excel("teaor_bin_summary.xlsx", usecols="J,K,M", header=None, engine="openpyxl")
# mutassunk 30 sort a közepéről is, mert lehet hogy header/üres blokkok vannak
print(df.head(20))
print(df.tail(20))

# Nézzük a bin oszlop egyedi értékeit (M oszlop = 2. index)
bins = df.iloc[:, 2].astype(str).str.strip().str.replace("–", "-", regex=False)
print("Unique bin-like values (first 50):")
print(sorted(set(bins))[:50])"""
