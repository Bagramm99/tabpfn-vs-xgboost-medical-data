"""Alle Experimente. Jede Funktion gibt ein DataFrame in 'langem' Format zurück (eine Zeile pro Modell und Lauf)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

from .evaluation import classification_metrics, fit_predict_timed
from .models import make_models


# ============================================================ Klassifikation
def _run_models(Xtr, ytr, Xte, yte, seed, extra):
    rows = []
    for name, m in make_models(seed).items():
        p, secs = fit_predict_timed(m, Xtr, ytr, Xte)
        rows.append({**extra, "Modell": name, **classification_metrics(yte, p), "Zeit_s": secs})
    return rows


def cv_comparison(X, y, label, n_seeds=5, n_splits=5):
    """Experiment 1: wiederholte stratifizierte Kreuzvalidierung (Hauptvergleich)."""
    rows = []
    for seed in range(n_seeds):
        for fold, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed).split(X, y)):
            rows += _run_models(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], seed, {"Setting": label, "Seed": seed, "Fold": fold})
    return pd.DataFrame(rows)


def learning_curve(X, y, sizes, n_seeds=10):
    """Experiment 2: kleine Stichproben (feste Testmenge 30 %, Trainingsgröße n)."""
    rows = []
    for n in sizes:
        for seed in range(n_seeds):
            try:
                tr, te = next(StratifiedShuffleSplit(1, train_size=n, test_size=0.3, random_state=seed).split(X, y))
            except ValueError:
                continue
            rows += _run_models(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te], seed, {"n_train": n, "Seed": seed})
    return pd.DataFrame(rows)


def short_fat(X, y, n_train, feature_counts, n_seeds=10):
    """Experiment 3 (Simulation): echte Merkmale + Rauschmerkmale -> p wächst, n bleibt klein."""
    rows = []
    for p in feature_counts:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            extra = max(0, p - X.shape[1])
            noise = pd.DataFrame(rng.normal(size=(len(X), extra)), columns=[f"noise_{i}" for i in range(extra)], index=X.index)
            Xn = pd.concat([X, noise], axis=1)
            tr, te = next(StratifiedShuffleSplit(1, train_size=n_train, test_size=0.3, random_state=seed).split(Xn, y))
            rows += _run_models(Xn.iloc[tr], y.iloc[tr], Xn.iloc[te], y.iloc[te], seed, {"n_features": Xn.shape[1], "Seed": seed})
    return pd.DataFrame(rows)

