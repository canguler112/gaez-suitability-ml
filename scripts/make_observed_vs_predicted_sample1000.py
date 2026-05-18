from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"C:\Users\cangu\OneDrive\Desktop\Agriculture")

PRED_CSV = PROJECT_DIR / "outputs" / "analysis" / "xgb_rs04_block2p0_cv_predictions.csv"

OUT_DIR = PROJECT_DIR / "outputs" / "analysis" / "xgb_rs04_block2p0_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "observed_vs_predicted_sample_1000.csv"

print("Reading prediction file...")
df = pd.read_csv(PRED_CSV)

df["observed"] = pd.to_numeric(df["observed"], errors="coerce")
df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
df = df.dropna(subset=["observed", "predicted"])

sample = df[["observed", "predicted"]].sample(n=1000, random_state=42)

sample.to_csv(OUT_CSV, index=False)

print("DONE")
print("Saved sample CSV to:")
print(OUT_CSV)