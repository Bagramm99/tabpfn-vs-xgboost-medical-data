"""Survival-Modelle mit einheitlicher Schnittstelle.

fit(X, time, event)
predict_risk(X)              -> Array, höher = früheres Ereignis (für C-Index)
predict_survival(X, times)   -> Matrix (n, len(times)) mit S(t|x)

Modelle: Cox (klassisch), Random Survival Forest (klassisch), DiscreteTimeTFM (Foundation-Model-Ansatz).
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv


def _surv(time, event):
    return Surv.from_arrays(event=np.asarray(event, bool), time=np.asarray(time, float))


class _SksurvBase:
    """Gemeinsame Vorverarbeitung (Imputation, Skalierung) für sksurv-Modelle."""
    def _prep_fit(self, X):
        self.imp_ = SimpleImputer(strategy="median").fit(X)
        self.sc_ = StandardScaler().fit(self.imp_.transform(X))
        return self.sc_.transform(self.imp_.transform(X))
    def _prep(self, X):
        return self.sc_.transform(self.imp_.transform(X))
    def predict_risk(self, X):
        return self.model_.predict(self._prep(X))
    def predict_survival(self, X, times):
        fns = self.model_.predict_survival_function(self._prep(X))
        return np.vstack([[fn(t) for t in times] for fn in fns])


class CoxModel(_SksurvBase):
    name = "Cox"
    def __init__(self, alpha=0.1, seed=0):
        self.alpha, self.seed = alpha, seed
    def fit(self, X, time, event):
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha).fit(self._prep_fit(X), _surv(time, event))
        return self


class RSFModel(_SksurvBase):
    name = "RSF"
    def __init__(self, n_estimators=300, min_samples_leaf=5, seed=0):
        self.n_estimators, self.min_samples_leaf, self.seed = n_estimators, min_samples_leaf, seed
    def fit(self, X, time, event):
        self.model_ = RandomSurvivalForest(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                           n_jobs=-1, random_state=self.seed).fit(self._prep_fit(X), _surv(time, event))
        return self


class DiscreteTimeTFM:
    """Foundation-Model-Ansatz: Survival als Klassifikation (Diskrete-Zeit-Hazard, Person-Period-Format).

    Idee (angelehnt an die Umformulierung von Survival als Klassifikation, vgl. Kim et al. 2026):
    Zeitachse in K Intervalle teilen. Jeder Patient erzeugt eine Zeile pro durchlaufenem Intervall
    (Merkmale + Intervallindex); Ziel = Ereignis in diesem Intervall. Ein Klassifikator (z. B. TabPFN)
    schätzt den Hazard h_k(x); S(t_k|x) = prod_{j<=k} (1 - h_j(x)).
    HINWEIS: Eigene Umsetzung des Prinzips, NICHT der Originalcode der Autoren. Vor dem Zitieren mit
    dem Paper abgleichen.

    classifier_factory: Funktion seed -> sklearn-Klassifikator (TabPFNClassifier, XGB, ...).
    max_rows begrenzt die Kontextgröße (TabPFN-Limit).
    """
    def __init__(self, classifier_factory, name="TFM-DT", n_bins=10, max_rows=8000, seed=0):
        self.factory, self.name, self.n_bins, self.max_rows, self.seed = classifier_factory, name, n_bins, max_rows, seed

    def _expand(self, X, time=None, event=None):
        """Person-Period-Tabelle. Training: nur bis Ereignis/Zensur. Vorhersage: alle Intervalle."""
        rows, ys = [], []
        K = len(self.edges_) - 1
        Xv = X.reset_index(drop=True)
        for i in range(len(Xv)):
            if time is None:
                last = K
            else:
                last = min(int(np.searchsorted(self.edges_, time[i], side="left")), K)  # Intervall von time[i]
                last = max(last, 1)
            for k in range(1, last + 1):
                rows.append((i, k))
                if time is not None:
                    ys.append(int(bool(event[i]) and k == last))
        idx = np.array([r[0] for r in rows]); ks = np.array([r[1] for r in rows])
        Z = Xv.iloc[idx].reset_index(drop=True).copy()
        Z["_interval"] = ks
        return Z, np.array(ys), idx, ks

    def fit(self, X, time, event):
        time = np.asarray(time, float); event = np.asarray(event, bool)
        qs = np.quantile(time[event] if event.sum() >= self.n_bins else time, np.linspace(0, 1, self.n_bins + 1))
        self.edges_ = np.unique(qs)                                   # Intervallgrenzen (Quantile der Ereigniszeiten)
        self.edges_[0] = min(self.edges_[0], time.min()) - 1e-9
        Z, y, _, _ = self._expand(X, time, event)
        if len(Z) > self.max_rows:                                    # Kontext begrenzen, Ereignisse behalten
            rng = np.random.default_rng(self.seed)
            pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
            keep = np.concatenate([pos, rng.choice(neg, size=max(self.max_rows - len(pos), 1), replace=False)])
            Z, y = Z.iloc[keep], y[keep]
        self.clf_ = self.factory(self.seed)
        self.clf_.fit(Z.fillna(Z.median(numeric_only=True)), y)
        self.medians_ = Z.median(numeric_only=True)
        return self

    def _hazards(self, X):
        Z, _, idx, ks = self._expand(X)
        h = self.clf_.predict_proba(Z.fillna(self.medians_))[:, 1]
        K = len(self.edges_) - 1
        H = np.zeros((len(X), K)); H[idx, ks - 1] = h
        return H

    def _survival_grid(self, X):
        return np.cumprod(1 - self._hazards(X), axis=1)              # S am Ende jedes Intervalls

    def predict_risk(self, X):
        return 1 - self._survival_grid(X).mean(axis=1)               # mittleres Ausfallrisiko über alle Intervalle

    def predict_survival(self, X, times):
        S = self._survival_grid(X)
        ends = self.edges_[1:]
        out = np.ones((len(S), len(times)))
        for j, t in enumerate(times):
            k = np.searchsorted(ends, t, side="left")                # erstes Intervall, das t enthält
            out[:, j] = S[:, min(k, S.shape[1] - 1)]
        return out


def make_survival_models(seed, use_tabpfn=True):
    models = {"Cox": CoxModel(seed=seed), "RSF": RSFModel(seed=seed)}
    if use_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            models["TabPFN-DT"] = DiscreteTimeTFM(lambda s: TabPFNClassifier(random_state=s), "TabPFN-DT", seed=seed)
        except Exception as e:
            from .models import _note
            _note(f"TabPFN nicht verfügbar ({e}).")
    # Kontrolle: derselbe Diskrete-Zeit-Ansatz mit klassischem Klassifikator (trennt Ansatz- von Modelleffekt)
    from xgboost import XGBClassifier
    models["XGB-DT"] = DiscreteTimeTFM(lambda s: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                                              eval_metric="logloss", random_state=s, n_jobs=-1),
                                       "XGB-DT", seed=seed)
    return models
