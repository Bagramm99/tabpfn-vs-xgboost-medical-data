"""Abbildungen für die Arbeit (aus den CSV-Ergebnissen)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def line_with_band(df, x, y, title, outfile, ylabel=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in df.groupby("Modell"):
        s = g.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.plot(s[x], s["mean"], marker="o", label=name)
        ax.fill_between(s[x], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15)
    ax.set_xlabel(x); ax.set_ylabel(ylabel or y); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=200); plt.close(fig)

def bar_with_error(df, y, title, outfile):
    s = df.groupby("Modell")[y].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(s.index, s["mean"], xerr=s["std"], capsize=3)
    ax.set_xlabel(y); ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=200); plt.close(fig)

def latex_table(df, y_cols, outfile, by=("Modell",)):
    """Mittelwert ± Std als LaTeX-Tabelle für die Arbeit."""
    g = df.groupby(list(by))[y_cols].agg(["mean", "std"])
    out = pd.DataFrame({c: g[(c, "mean")].round(3).astype(str) + " ± " + g[(c, "std")].round(3).astype(str) for c in y_cols})
    out.to_latex(outfile, escape=True)
