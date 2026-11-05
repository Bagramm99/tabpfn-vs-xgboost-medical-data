"""Klassifikations-Experimente.

Beispiele (aus dem Projektordner):
  python -m scripts.run_klassifikation --dataset breast_cancer
  python -m scripts.run_klassifikation --dataset csv --path data/raw/leber.csv --target Diagnose --name leber
"""
import argparse
import pandas as pd
from src import config, data_loading as dl, experiments as ex, plots

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "csv", "openml"])
ap.add_argument("--path"); ap.add_argument("--target"); ap.add_argument("--name", default=None)
ap.add_argument("--sep", default=","); ap.add_argument("--decimal", default="."); ap.add_argument("--encoding", default="utf-8")
ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
a = ap.parse_args()

if a.dataset == "breast_cancer":
    X, y = dl.load_breast_cancer_wdbc(); name = a.name or "breast_cancer"
elif a.dataset == "openml":
    X, y = dl.load_openml_classification(a.target); name = a.name or a.target
else:
    X, y = dl.load_csv_classification(a.path, a.target, a.sep, a.decimal, a.encoding); name = a.name or "csv"

out = config.RESULTS / name; out.mkdir(parents=True, exist_ok=True)
print(dl.describe(X, y, name), f"Positivanteil={y.mean():.2f}")

r1 = ex.cv_comparison(X, y, "alle Merkmale", a.seeds); r1.to_csv(out / "1_cv.csv", index=False)
print(r1.groupby("Modell")[["AUROC", "PR-AUC", "Brier", "Zeit_s"]].agg(["mean", "std"]).round(3))

sizes = [s for s in config.TRAIN_SIZES if s < len(X) * 0.7]
r2 = ex.learning_curve(X, y, sizes, a.seeds * 2); r2.to_csv(out / "2_lernkurve.csv", index=False)
plots.line_with_band(r2, "n_train", "AUROC", f"Lernkurve – {name}", config.FIGURES / f"{name}_lernkurve.png")

r3 = ex.short_fat(X, y, min(config.SHORT_FAT_N_TRAIN, int(len(X) * 0.5)), [X.shape[1]] + config.FEATURE_COUNTS, a.seeds * 2)
r3.to_csv(out / "3_short_fat.csv", index=False)
plots.line_with_band(r3, "n_features", "AUROC", f"Short-fat (simuliert) – {name}", config.FIGURES / f"{name}_short_fat.png")

plots.latex_table(r1, ["AUROC", "PR-AUC", "Brier", "Zeit_s"], out / "tabelle_cv.tex")
print("Ergebnisse in", out)
