"""Survival-Experimente.

Beispiele:
  python -m scripts.run_survival --dataset whas500
  python -m scripts.run_survival --dataset influenza --path data/raw/influenza_rsv_ext.csv
  python -m scripts.run_survival --dataset csv --path data/raw/x.csv --time-col T --event-col E
"""
import argparse
from src import config, data_loading as dl, experiments as ex, plots

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="whas500", choices=["whas500", "influenza", "csv"])
ap.add_argument("--path"); ap.add_argument("--time-col", default="Dauer"); ap.add_argument("--event-col", default="had_event")
ap.add_argument("--seeds", type=int, default=config.N_SEEDS)
a = ap.parse_args()

if a.dataset == "whas500":
    X, t, e = dl.load_whas500(); name = "whas500"
elif a.dataset == "influenza":
    X, t, e = dl.load_influenza_rsv(a.path, a.time_col, a.event_col); name = "influenza_rsv"
else:
    X, t, e = dl.load_csv_survival(a.path, a.time_col, a.event_col); name = "csv_survival"

out = config.RESULTS / name; out.mkdir(parents=True, exist_ok=True)
print(dl.describe(X, name=name), f"Ereignisse={int(e.sum())}/{len(e)} ({e.mean():.0%}), Zensierung={1 - e.mean():.0%}")

r1 = ex.survival_cv(X, t, e, "alle Merkmale", a.seeds); r1.to_csv(out / "5_survival_cv.csv", index=False)
print(r1.groupby("Modell")[["C-Index", "IBS", "AUC(t_med)"]].agg(["mean", "std"]).round(3))
plots.bar_with_error(r1, "C-Index", f"C-Index – {name}", config.FIGURES / f"{name}_cindex.png")

sizes = [s for s in [50, 100, 200, 300] if s < len(X) * 0.7]
r2 = ex.survival_learning_curve(X, t, e, sizes, a.seeds * 2); r2.to_csv(out / "6_survival_lernkurve.csv", index=False)
plots.line_with_band(r2, "n_train", "C-Index", f"Survival-Lernkurve – {name}", config.FIGURES / f"{name}_surv_lernkurve.png")
plots.latex_table(r1, [c for c in ["C-Index", "IBS", "AUC(t_med)"] if c in r1], out / "tabelle_survival.tex")
print("Ergebnisse in", out)
