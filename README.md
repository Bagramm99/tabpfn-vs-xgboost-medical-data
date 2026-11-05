# Masterprojekt: Tabular Foundation Models in der medizinischen Datenanalyse
Klassifikation auf kleinen und hochdimensionalen Datensätzen (F1 Hauptvergleich, F2 kleine Stichproben, F3 Short-fat).
Betreuer: Prof. Dr. Frank Klawonn (Ostfalia).

## Installation
    python -m venv .venv  (oder PyCharm-Interpreter anlegen)
    pip install -r requirements.txt
TabPFN lädt beim ersten Start Gewichte (Internet, ggf. Hugging-Face-Login).

## Ausführen (aus dem Projektordner)
    python -m scripts.run_klassifikation --dataset breast_cancer
    python -m scripts.run_klassifikation --dataset csv --path data/raw/leber.csv --target Diagnose --name leber
Schnelltest: `--seeds 1`. Ergebnisse: `ergebnisse/<datensatz>/`, Abbildungen: `figures/`.

## Struktur
src/config.py, data_loading.py, models.py, evaluation.py, experiments.py (Exp. 1-3), plots.py; scripts/run_klassifikation.py;
archiv_masterarbeit/ (Survival-Code für später).

## Regeln
Rohdaten in data/raw nie ändern (nicht ins Git). Vorverarbeitung nur innerhalb der Folds.
