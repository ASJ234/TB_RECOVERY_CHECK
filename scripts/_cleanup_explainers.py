"""Clean ALL cached SHAP explainers so they are rebuilt fresh with correct settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPLAINERS_DIR = ROOT / "models" / "explainers"

deleted = []
for f in EXPLAINERS_DIR.rglob("*.joblib"):
    f.unlink()
    deleted.append(str(f.relative_to(ROOT)))
    print(f"  Deleted: {f.relative_to(ROOT)}")

if not deleted:
    print("  No cached explainer files found.")
else:
    print(f"\n  {len(deleted)} file(s) removed. All explainers will be rebuilt fresh on next run.")
