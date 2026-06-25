import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"


TARGET_PREFIXES = ("TARGET_", "Sample ID", "STUDY_ID", "CONTACT_NUMBER", "DATE", "INITIALS")


def _should_skip(col: str, exclude_cols: list = None) -> bool:
    if exclude_cols and col in exclude_cols:
        return True
    return col.startswith(TARGET_PREFIXES)


def generate_gaussian_drift(
    df: pd.DataFrame,
    noise_std: float = 0.5,
    random_state: int = 42,
    exclude_cols: list = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df_drifted = df.copy()
    for c in df.columns:
        if _should_skip(c, exclude_cols):
            continue
        if df[c].dtype in (np.float64, np.int64, float, int):
            noise = rng.normal(0, noise_std, size=len(df))
            df_drifted[c] = df[c].astype(float) + noise
    return df_drifted


def generate_category_swap_drift(
    df: pd.DataFrame,
    swap_pct: float = 0.3,
    random_state: int = 42,
    exclude_cols: list = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df_drifted = df.copy()
    for c in df.columns:
        if _should_skip(c, exclude_cols):
            continue
        if df[c].dtype == object:
            unique_vals = df[c].dropna().unique()
            if len(unique_vals) >= 2:
                mask = rng.random(len(df)) < swap_pct
                shuffled = rng.permutation(unique_vals.tolist() * (len(df) // len(unique_vals) + 1))[:mask.sum()]
                df_drifted.loc[mask, c] = shuffled
    return df_drifted


def generate_combined_drift(
    df: pd.DataFrame,
    noise_std: float = 1.0,
    swap_pct: float = 0.4,
    random_state: int = 42,
    exclude_cols: list = None,
) -> pd.DataFrame:
    df_temp = generate_gaussian_drift(df, noise_std=noise_std, random_state=random_state, exclude_cols=exclude_cols)
    return generate_category_swap_drift(df_temp, swap_pct=swap_pct, random_state=random_state + 1, exclude_cols=exclude_cols)


def generate_all_synthetic_variants(
    source_csv: str,
    output_prefix: str = "",
    random_state: int = 42,
) -> dict:
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    source_path = DATA_DIR / "cleaned" / source_csv
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")
    df = pd.read_csv(source_path)

    variants = {}

    gaussian = generate_gaussian_drift(df, random_state=random_state)
    gaussian_path = SYNTHETIC_DIR / f"{output_prefix or source_csv.replace('.csv', '')}_gaussian.csv"
    gaussian.to_csv(gaussian_path, index=False)
    variants["gaussian"] = str(gaussian_path)

    swap = generate_category_swap_drift(df, random_state=random_state)
    swap_path = SYNTHETIC_DIR / f"{output_prefix or source_csv.replace('.csv', '')}_swap.csv"
    swap.to_csv(swap_path, index=False)
    variants["swap"] = str(swap_path)

    combined = generate_combined_drift(df, random_state=random_state)
    combined_path = SYNTHETIC_DIR / f"{output_prefix or source_csv.replace('.csv', '')}_drifted.csv"
    combined.to_csv(combined_path, index=False)
    variants["combined"] = str(combined_path)

    print(f"  Synthetic variants generated in {SYNTHETIC_DIR}:")
    for name, path in variants.items():
        print(f"    {name}: {path}")
    return variants


if __name__ == "__main__":
    generate_all_synthetic_variants("aim1_patients_imputed.csv", output_prefix="aim1")
    generate_all_synthetic_variants("aim2_contacts_imputed.csv", output_prefix="aim2")
    generate_all_synthetic_variants("aim1_patients_strict.csv", output_prefix="aim1_strict")
