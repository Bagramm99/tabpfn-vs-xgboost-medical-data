"""Zentrale Einstellungen. Alle Skripte lesen von hier, damit Ergebnisse reproduzierbar bleiben."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "ergebnisse"
FIGURES = ROOT / "figures"

N_SEEDS = 5                      # Wiederholungen der Kreuzvalidierung
N_SPLITS = 5                     # Folds pro Wiederholung
TRAIN_SIZES = [20, 50, 100, 200, 300]       # Lernkurve
FEATURE_COUNTS = [30, 100, 300, 1000]       # short-fat (Simulation)
SHORT_FAT_N_TRAIN = 80
