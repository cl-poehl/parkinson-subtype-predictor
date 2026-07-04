"""Noise robustness analysis: perturbs patient features with Gaussian noise
and measures whether the class prediction changes.

Input: feats (DataFrame of all patients, for range estimation),
patient_idx (position in the DataFrame), models (dict label->model).

Output: list of rows with original P(Fast), range, and flip probability
per classifier."""
import numpy as np
import pandas as pd


def noise_sensitivity(feats, patient_idx, models,
                       n_perturbations=30, noise_sd_rel=0.10, seed=42):
    """List of diagnostic rows per classifier."""
    if patient_idx is None or feats is None or feats.empty:
        return []
    row = feats.iloc[patient_idx:patient_idx + 1].copy()

    sds = {}
    for col in feats.columns:
        col_vals = feats[col].dropna().values
        if col_vals.size == 0:
            sds[col] = 0.0
            continue
        sds[col] = float(noise_sd_rel * (col_vals.max() - col_vals.min()))

    rng = np.random.default_rng(seed)
    rows = []
    for clf_name, model in models.items():
        try:
            original_p = float(model.predict_proba(row.values)[0, 1])
        except Exception:
            continue
        perturbed = []
        for _ in range(n_perturbations):
            noisy = row.copy()
            for col in feats.columns:
                if pd.notna(noisy.iloc[0][col]) and sds[col] > 0:
                    noisy.iloc[0, noisy.columns.get_loc(col)] = (
                        row.iloc[0][col] + rng.normal(0, sds[col]))
            try:
                p = float(model.predict_proba(noisy.values)[0, 1])
            except Exception:
                continue
            perturbed.append(p)
        if not perturbed:
            continue
        perturbed = np.array(perturbed)
        original_class = 1 if original_p >= 0.5 else 0
        flip = float(((perturbed >= 0.5).astype(int) != original_class).mean())
        lo, hi = np.quantile(perturbed, [0.05, 0.95])
        rows.append({
            "Method": clf_name,
            "Original P(Fast)": f"{original_p*100:.1f}%",
            "P(Fast) range (5-95%)": f"{lo*100:.1f}% - {hi*100:.1f}%",
            "Flip probability": f"{flip*100:.1f}%",
        })
    return rows
