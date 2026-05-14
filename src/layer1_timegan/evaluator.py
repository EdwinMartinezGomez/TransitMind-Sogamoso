"""
TransitMind Sogamoso — TSTR Evaluator (Fase 4)
================================================
Evaluates synthetic data quality using the TSTR protocol.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.shared.constants import NUMERIC_FEATURES, TARGET_VARIABLE
from src.shared.logger import setup_logger
from src.shared.utils import ensure_dir, resolve_path

logger = setup_logger("evaluator")


class _SimpleLSTM(nn.Module):
    """Simple LSTM predictor for TSTR evaluation."""
    def __init__(self, n_features: int, hidden: int = 32, target_idx: int = 0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden, 1)
        self.target_idx = target_idx

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _SimpleClassifier(nn.Module):
    """Simple classifier for discriminative score."""
    def __init__(self, n_features: int, seq_len: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


def _prepare_sequences(df: pd.DataFrame, seq_len: int = 24, target_col: str = TARGET_VARIABLE):
    """Convert DataFrame to sequences for evaluation."""
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    values = df[features].values.astype(np.float32)
    target_idx = features.index(target_col) if target_col in features else 0
    seqs, targets = [], []
    for i in range(len(values) - seq_len):
        seqs.append(values[i:i + seq_len])
        targets.append(values[i + seq_len - 1, target_idx])
    return np.array(seqs), np.array(targets), target_idx


def _train_predictor(X, y, target_idx, n_features, epochs=50):
    """Train a simple LSTM predictor."""
    model = _SimpleLSTM(n_features, target_idx=target_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    X_t, y_t = torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def tstr_evaluation(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    target_col: str = TARGET_VARIABLE,
    seq_len: int = 24,
) -> Dict[str, float]:
    """
    TSTR (Train on Synthetic, Test on Real) evaluation.
    
    a) Train LSTM on SYNTHETIC, test on REAL
    b) Train LSTM on REAL, test on REAL (baseline)
    c) Compare MAE, RMSE, R²
    
    TSTR-score = 1 - |MAE_synthetic - MAE_real| / MAE_real
    Target: >= 0.85
    """
    logger.info("tstr_evaluation_start")
    
    X_syn, y_syn, tidx = _prepare_sequences(synthetic_df, seq_len, target_col)
    X_real, y_real, _ = _prepare_sequences(real_df, seq_len, target_col)
    
    if len(X_syn) == 0 or len(X_real) == 0:
        logger.warning("insufficient_data_for_tstr")
        return {"tstr_score": 0.0, "mae_synthetic": 1.0, "mae_real": 0.0}
    
    n_features = X_syn.shape[2]
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )
    
    # Train on synthetic, test on real
    model_syn = _train_predictor(X_syn, y_syn, tidx, n_features)
    model_syn.eval()
    with torch.no_grad():
        pred_syn = model_syn(torch.FloatTensor(X_real_test)).numpy().flatten()
    
    # Train on real, test on real (baseline)
    model_real = _train_predictor(X_real_train, y_real_train, tidx, n_features)
    model_real.eval()
    with torch.no_grad():
        pred_real = model_real(torch.FloatTensor(X_real_test)).numpy().flatten()
    
    mae_syn = mean_absolute_error(y_real_test, pred_syn)
    mae_real = mean_absolute_error(y_real_test, pred_real)
    rmse_syn = float(np.sqrt(mean_squared_error(y_real_test, pred_syn)))
    rmse_real = float(np.sqrt(mean_squared_error(y_real_test, pred_real)))
    r2_syn = r2_score(y_real_test, pred_syn)
    r2_real = r2_score(y_real_test, pred_real)
    
    tstr_score = max(0.0, 1.0 - abs(mae_syn - mae_real) / max(mae_real, 1e-8))
    
    metrics = {
        "tstr_score": round(float(tstr_score), 4),
        "mae_synthetic": round(float(mae_syn), 4),
        "mae_real": round(float(mae_real), 4),
        "rmse_synthetic": round(rmse_syn, 4),
        "rmse_real": round(rmse_real, 4),
        "r2_synthetic": round(float(r2_syn), 4),
        "r2_real": round(float(r2_real), 4),
    }
    
    logger.info("tstr_evaluation_complete", **metrics)
    return metrics


def discriminative_score(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    seq_len: int = 24,
) -> Dict[str, float]:
    """
    Train a classifier to distinguish real vs synthetic.
    If accuracy ≈ 0.5 → data is indistinguishable (ideal).
    Target: accuracy < 0.55
    """
    logger.info("discriminative_score_start")
    
    features = [c for c in NUMERIC_FEATURES if c in synthetic_df.columns and c in real_df.columns]
    
    def make_seqs(df):
        v = df[features].values.astype(np.float32)
        return np.array([v[i:i+seq_len] for i in range(len(v) - seq_len)])
    
    syn_seqs = make_seqs(synthetic_df)
    real_seqs = make_seqs(real_df)
    
    min_n = min(len(syn_seqs), len(real_seqs))
    if min_n < 10:
        return {"accuracy": 0.5, "auc_roc": 0.5}
    
    syn_seqs = syn_seqs[:min_n]
    real_seqs = real_seqs[:min_n]
    
    X = np.concatenate([real_seqs, syn_seqs])
    y = np.concatenate([np.ones(min_n), np.zeros(min_n)])
    
    idx = np.random.RandomState(42).permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = _SimpleClassifier(len(features), seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(30):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).numpy().flatten()
    
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    try:
        auc = roc_auc_score(y_test, preds)
    except ValueError:
        auc = 0.5
    
    result = {"accuracy": round(float(acc), 4), "auc_roc": round(float(auc), 4)}
    logger.info("discriminative_score_complete", **result)
    return result


def visualize_distributions(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Generate comparison visualizations: t-SNE, PCA, KDE, autocorrelation."""
    if output_dir is None:
        output_dir = str(resolve_path("experiments/notebooks/outputs"))
    ensure_dir(output_dir)
    
    features = [c for c in NUMERIC_FEATURES if c in synthetic_df.columns and c in real_df.columns]
    syn_vals = synthetic_df[features].values.astype(np.float32)
    real_vals = real_df[features].values.astype(np.float32)
    saved = []
    
    # t-SNE
    try:
        n = min(500, len(syn_vals), len(real_vals))
        combined = np.vstack([real_vals[:n], syn_vals[:n]])
        labels = ["Real"] * n + ["Synthetic"] * n
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
        embedded = tsne.fit_transform(combined)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(embedded[:n, 0], embedded[:n, 1], alpha=0.5, label="Real", s=10)
        ax.scatter(embedded[n:, 0], embedded[n:, 1], alpha=0.5, label="Synthetic", s=10)
        ax.legend()
        ax.set_title("t-SNE: Real vs Synthetic Traffic Data")
        path = str(Path(output_dir) / "tsne_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        logger.warning("tsne_failed", error=str(e))
    
    # PCA
    try:
        pca = PCA(n_components=2)
        n = min(1000, len(syn_vals), len(real_vals))
        combined = np.vstack([real_vals[:n], syn_vals[:n]])
        projected = pca.fit_transform(combined)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(projected[:n, 0], projected[:n, 1], alpha=0.5, label="Real", s=10)
        ax.scatter(projected[n:, 0], projected[n:, 1], alpha=0.5, label="Synthetic", s=10)
        ax.legend()
        ax.set_title("PCA: Real vs Synthetic Traffic Data")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        path = str(Path(output_dir) / "pca_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    except Exception as e:
        logger.warning("pca_failed", error=str(e))
    
    # KDE per variable
    try:
        plot_features = ["vehicle_flow", "avg_speed_kmh", "congestion_level"]
        plot_features = [f for f in plot_features if f in features]
        if plot_features:
            fig, axes = plt.subplots(1, len(plot_features), figsize=(5 * len(plot_features), 4))
            if len(plot_features) == 1:
                axes = [axes]
            for ax, feat in zip(axes, plot_features):
                sns.kdeplot(real_df[feat].dropna(), ax=ax, label="Real", fill=True, alpha=0.3)
                sns.kdeplot(synthetic_df[feat].dropna(), ax=ax, label="Synthetic", fill=True, alpha=0.3)
                ax.set_title(feat)
                ax.legend()
            fig.suptitle("Distribution Comparison (KDE)")
            fig.tight_layout()
            path = str(Path(output_dir) / "kde_distributions.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    except Exception as e:
        logger.warning("kde_failed", error=str(e))
    
    logger.info("visualizations_saved", count=len(saved))
    return saved


def compute_fidelity_metrics(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute statistical fidelity metrics: JS divergence, correlation diff, descriptive stats."""
    features = [c for c in NUMERIC_FEATURES if c in synthetic_df.columns and c in real_df.columns]
    
    # JS Divergence per variable
    js_scores = {}
    for feat in features:
        syn_vals = synthetic_df[feat].dropna().values
        real_vals = real_df[feat].dropna().values
        bins = np.linspace(min(real_vals.min(), syn_vals.min()), max(real_vals.max(), syn_vals.max()), 50)
        hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
        hist_syn, _ = np.histogram(syn_vals, bins=bins, density=True)
        hist_real = hist_real + 1e-10
        hist_syn = hist_syn + 1e-10
        hist_real /= hist_real.sum()
        hist_syn /= hist_syn.sum()
        js_scores[feat] = round(float(jensenshannon(hist_real, hist_syn) ** 2), 6)
    
    js_avg = round(float(np.mean(list(js_scores.values()))), 6)
    
    # Correlation matrix difference
    corr_real = real_df[features].corr().values
    corr_syn = synthetic_df[features].corr().values
    corr_diff = round(float(np.mean(np.abs(corr_real - corr_syn))), 6)
    
    # Descriptive stats comparison
    stats_comparison = {}
    for feat in features:
        stats_comparison[feat] = {
            "mean_diff": round(abs(float(real_df[feat].mean() - synthetic_df[feat].mean())), 6),
            "std_diff": round(abs(float(real_df[feat].std() - synthetic_df[feat].std())), 6),
        }
    
    result = {
        "js_divergence": js_scores,
        "js_divergence_avg": js_avg,
        "correlation_diff": corr_diff,
        "stats_comparison": stats_comparison,
    }
    
    logger.info("fidelity_metrics_computed", js_avg=js_avg, corr_diff=corr_diff)
    return result


def generate_evaluation_report(all_metrics: Dict[str, Any]) -> str:
    """Generate a markdown evaluation report and save as model_card.md."""
    tstr = all_metrics.get("tstr", {})
    disc = all_metrics.get("discriminative", {})
    fidelity = all_metrics.get("fidelity", {})
    
    tstr_score = tstr.get("tstr_score", 0)
    disc_acc = disc.get("accuracy", 0.5)
    js_avg = fidelity.get("js_divergence_avg", 0)
    corr_diff = fidelity.get("correlation_diff", 0)
    
    passed = tstr_score >= 0.80 and disc_acc <= 0.60 and js_avg <= 0.15 and corr_diff <= 0.10
    status = "✅ PASSED" if passed else "❌ NEEDS IMPROVEMENT"
    
    report = f"""# TransitMind Sogamoso — TimeGAN Model Card

## Resumen Ejecutivo
- **Estado**: {status}
- **TSTR-Score**: {tstr_score:.4f} (meta: ≥ 0.85)
- **Discriminative Accuracy**: {disc_acc:.4f} (meta: ≤ 0.60)
- **JS Divergence Avg**: {js_avg:.6f} (meta: ≤ 0.15)
- **Correlation Diff**: {corr_diff:.6f} (meta: ≤ 0.10)

## Métricas TSTR
| Métrica | Sintético | Real |
|---------|-----------|------|
| MAE | {tstr.get('mae_synthetic', 'N/A')} | {tstr.get('mae_real', 'N/A')} |
| RMSE | {tstr.get('rmse_synthetic', 'N/A')} | {tstr.get('rmse_real', 'N/A')} |
| R² | {tstr.get('r2_synthetic', 'N/A')} | {tstr.get('r2_real', 'N/A')} |

## Discriminative Score
- **Accuracy**: {disc_acc:.4f}
- **AUC-ROC**: {disc.get('auc_roc', 'N/A')}

## Criterios de Aceptación
| Métrica | Valor | Umbral Mín | Umbral Ideal | Estado |
|---------|-------|------------|--------------|--------|
| TSTR-score | {tstr_score:.4f} | ≥ 0.80 | ≥ 0.90 | {'✅' if tstr_score >= 0.80 else '❌'} |
| Disc. Accuracy | {disc_acc:.4f} | ≤ 0.60 | ≤ 0.52 | {'✅' if disc_acc <= 0.60 else '❌'} |
| JS Divergence | {js_avg:.6f} | ≤ 0.15 | ≤ 0.08 | {'✅' if js_avg <= 0.15 else '❌'} |
| Correlation Diff | {corr_diff:.6f} | ≤ 0.10 | ≤ 0.05 | {'✅' if corr_diff <= 0.10 else '❌'} |

## Recomendaciones
{"Los datos sintéticos cumplen los criterios mínimos de calidad." if passed else "Se recomienda ajustar hiperparámetros y re-entrenar el modelo."}

---
*Generado automáticamente por TransitMind Sogamoso — TSTR Evaluator*
"""
    
    path = resolve_path("models/timegan/model_card.md")
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info("evaluation_report_saved", path=str(path), passed=passed)
    return report
