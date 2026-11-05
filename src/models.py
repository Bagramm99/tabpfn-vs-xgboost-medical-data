"""Klassifikationsmodelle: Baselines + Tabular Foundation Models."""
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
_WARNED = set()

def _note(msg):
    if msg not in _WARNED:
        print(f"[Hinweis] {msg}")
        _WARNED.add(msg)

def make_models(seed, use_tabpfn=True, use_tabicl=False):
    models = {
        "LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegressionCV(Cs=10, cv=3, max_iter=5000)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 eval_metric="logloss", random_state=seed, n_jobs=-1),
    }
    if use_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            models["TabPFN"] = TabPFNClassifier(random_state=seed)
        except Exception as e:
            _note(f"TabPFN nicht verfügbar ({e}).")
    if use_tabicl:
        try:
            from tabicl import TabICLClassifier
            models["TabICL"] = TabICLClassifier(random_state=seed)
        except Exception as e:
            _note(f"TabICL nicht verfügbar ({e}).")
    return models
