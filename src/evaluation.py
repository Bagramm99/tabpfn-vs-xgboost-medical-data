"""Metriken für Klassifikation. (Survival-Metriken: archiv_masterarbeit)"""
import time as _time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix


def classification_metrics(y_true, p, threshold=0.5):
    tn, fp, fn, tp = confusion_matrix(y_true, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return {"AUROC": roc_auc_score(y_true, p), "PR-AUC": average_precision_score(y_true, p),
            "Brier": brier_score_loss(y_true, p),
            "Sensitivität": tp / max(tp + fn, 1), "Spezifität": tn / max(tn + fp, 1)}


def fit_predict_timed(model, Xtr, ytr, Xte):
    t0 = _time.time()
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return p, _time.time() - t0
