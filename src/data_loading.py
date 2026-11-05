"""Laden und Bereinigen aller Datensätze.

Klassifikation -> (X: DataFrame, y: Series mit 0/1)
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------------- Klassifikation
def load_breast_cancer_wdbc():
    """UCI Breast Cancer Wisconsin (Diagnostic): 569 x 30, Ziel 1 = benigne, 0 = maligne. Kein Alter vorhanden."""
    d = load_breast_cancer(as_frame=True)
    return d.data, pd.Series(d.target, name="target")

def load_csv_classification(path, target, sep=",", decimal=".", encoding="utf-8"):
    """Beliebige CSV (z. B. reflimR-Leberdaten, aus R mit write.csv exportiert). Nur numerische Merkmale."""
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding)
    y = pd.Series(pd.factorize(df[target])[0], name="target")
    X = df.drop(columns=[target]).select_dtypes("number").reset_index(drop=True)
    return X, y

def load_openml_classification(name, target=None):
    """Optional: Datensatz von OpenML (Internet nötig), z. B. 'diabetes' (Pima)."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name=name, as_frame=True, parser="auto")
    X = d.data.select_dtypes("number").reset_index(drop=True)
    y = pd.Series(pd.factorize(d.target)[0], name="target")
    return X, y

# ---------------------------------------------------------------- Hilfsfunktion
def describe(X, y=None, name=""):
    """Eine Zeile für die Datensatztabelle der Arbeit."""
    miss = float(X.isna().mean().mean())
    return {"Datensatz": name, "n": len(X), "p": X.shape[1], "fehlende_Werte_Anteil": round(miss, 3)}
