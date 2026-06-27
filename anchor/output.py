from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")


def _stage_path(stage: Any) -> Path:
    return Path(getattr(stage, "path", stage))


def _read_stage_config(stage: Any) -> dict[str, Any]:
    path = _stage_path(stage)
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def predict_teacher(stage: str | Path | Any) -> tuple[pd.Series, pd.DataFrame | None]:
    """Read teacher predictions and soft probabilities from a canonical stage directory."""
    import anndata as ad

    path = _stage_path(stage)
    config = _read_stage_config(path)
    pred_col = str(config.get("pred_col", f"pred_{path.name}"))
    results_h5ad = path / "results.h5ad"
    if not results_h5ad.exists():
        raise FileNotFoundError(f"Missing teacher results h5ad: {results_h5ad}")
    adata = ad.read_h5ad(results_h5ad, backed=None)
    if pred_col not in adata.obs:
        raise KeyError(f"Prediction column {pred_col!r} not found in {results_h5ad}")
    pred = adata.obs[pred_col].astype(str).copy()
    soft_path = path / "soft_probs_all_cells.csv"
    soft = pd.read_csv(soft_path, index_col=0) if soft_path.exists() else None
    if soft is not None:
        soft.index = soft.index.astype(str)
    return pred, soft


def get_teacher_latent(stage: str | Path | Any) -> pd.DataFrame:
    """Read teacher latent embedding from a canonical stage directory."""
    import anndata as ad

    path = _stage_path(stage)
    config = _read_stage_config(path)
    latent_key = str(config.get("latent_key", f"X_{path.name}"))
    results_h5ad = path / "results.h5ad"
    if not results_h5ad.exists():
        raise FileNotFoundError(f"Missing teacher results h5ad: {results_h5ad}")
    adata = ad.read_h5ad(results_h5ad, backed=None)
    if latent_key not in adata.obsm:
        raise KeyError(f"Latent key {latent_key!r} not found in {results_h5ad}")
    return pd.DataFrame(np.asarray(adata.obsm[latent_key]), index=adata.obs_names.astype(str))


def predict_student(stage: str | Path | Any) -> tuple[pd.Series, pd.DataFrame | None]:
    """Read student predictions and soft probabilities from a canonical student directory."""
    import anndata as ad

    path = _stage_path(stage)
    results_h5ad = path / "student_results.h5ad"
    if not results_h5ad.exists():
        raise FileNotFoundError(f"Missing student results h5ad: {results_h5ad}")
    adata = ad.read_h5ad(results_h5ad, backed=None)
    if "student_pred_label" not in adata.obs:
        raise KeyError(f"Prediction column 'student_pred_label' not found in {results_h5ad}")
    pred = adata.obs["student_pred_label"].astype(str).copy()
    soft_path = path / "student_soft_probs.csv"
    soft = pd.read_csv(soft_path, index_col=0) if soft_path.exists() else None
    if soft is not None:
        soft.index = soft.index.astype(str)
    return pred, soft


def get_student_latent(stage: str | Path | Any, *, key: str = "u_student") -> pd.DataFrame:
    """Read student latent/prototype embedding from a canonical student directory."""
    import anndata as ad

    path = _stage_path(stage)
    results_h5ad = path / "student_results.h5ad"
    if not results_h5ad.exists():
        raise FileNotFoundError(f"Missing student results h5ad: {results_h5ad}")
    adata = ad.read_h5ad(results_h5ad, backed=None)
    if key not in adata.obsm:
        raise KeyError(f"Student embedding key {key!r} not found in {results_h5ad}")
    return pd.DataFrame(np.asarray(adata.obsm[key]), index=adata.obs_names.astype(str))


def history_to_dataframe(history: Any) -> pd.DataFrame:
    if history is None:
        return pd.DataFrame()
    if isinstance(history, pd.DataFrame):
        return history.copy()
    if isinstance(history, pd.Series):
        return history.to_frame().T
    if isinstance(history, Mapping):
        try:
            return pd.DataFrame(history)
        except ValueError:
            return pd.Series(history).to_frame().T
    try:
        return pd.DataFrame(history)
    except Exception:
        return pd.DataFrame({"history_repr": [repr(history)]})


def add_prediction_outputs(
    adata: ad.AnnData,
    *,
    pred: pd.Series,
    soft: pd.DataFrame,
    latent: np.ndarray,
    pred_col: str,
    latent_key: str,
    confidence_col: str,
    entropy_col: str,
    correct_col: str,
    label_col: str = "true_label",
    reference_name: str = "reference",
    query_name: str = "query",
) -> pd.Series:
    adata.obsm[latent_key] = latent.astype(np.float32)
    adata.obs[pred_col] = pred.reindex(adata.obs_names.astype(str)).astype(str).to_numpy()
    adata.obs[confidence_col] = soft.max(axis=1).reindex(adata.obs_names.astype(str)).astype(float).to_numpy()
    entropy = -(soft * np.log(np.clip(soft, 1e-12, None))).sum(axis=1)
    adata.obs[entropy_col] = entropy.reindex(adata.obs_names.astype(str)).astype(float).to_numpy()
    adata.obs[correct_col] = np.where(
        adata.obs[pred_col].astype(str).eq(adata.obs[label_col].astype(str)),
        "correct",
        "incorrect",
    )
    adata.obs.loc[adata.obs["ref_query_col"].astype(str).eq(reference_name), correct_col] = "reference"
    return evaluate_query_summary(adata.obs, pred_col=pred_col, label_col=label_col, query_name=query_name)


def evaluate_query_summary(
    obs: pd.DataFrame,
    *,
    pred_col: str,
    label_col: str = "true_label",
    query_name: str = "query",
) -> pd.Series:
    mask = obs["ref_query_col"].astype(str).eq(query_name)
    y_true = obs.loc[mask, label_col].astype(str)
    y_pred = obs.loc[mask, pred_col].astype(str)
    return pd.Series(
        {
            "query_accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
            "query_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else np.nan,
            "query_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) else np.nan,
            "n_query": int(len(y_true)),
        }
    )


def query_confusion_counts(
    obs: pd.DataFrame,
    pred_col: str,
    *,
    label_col: str = "true_label",
    query_name: str = "query",
) -> pd.DataFrame:
    mask = obs["ref_query_col"].astype(str).eq(query_name)
    y_true = obs.loc[mask, label_col].astype(str)
    y_pred = obs.loc[mask, pred_col].astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    return pd.crosstab(y_true, y_pred).reindex(index=labels, columns=labels, fill_value=0)


def plot_confusion_heatmap(counts: pd.DataFrame, *, title: str, out_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_norm = counts.astype(float).div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fig_w = max(8, min(24, 0.42 * max(1, counts.shape[1]) + 4))
    fig_h = max(6, min(24, 0.34 * max(1, counts.shape[0]) + 3))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        row_norm,
        cmap="Blues",
        vmin=0,
        vmax=max(1.0, float(row_norm.to_numpy().max(initial=0.0))),
        annot=counts,
        fmt="d",
        annot_kws={"fontsize": 5 if max(counts.shape or (1,)) > 24 else 6},
        cbar_kws={"label": "row-normalized fraction"},
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_student_confusion_heatmap(
    obs: pd.DataFrame,
    *,
    results_dir: Path,
    pred_col: str = "student_pred_label",
    label_col: str = "true_label",
    normalize_rows: bool = True,
    run_label: str | None = None,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(results_dir) / "confusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = obs[label_col].astype(str)
    y_pred = obs[pred_col].astype(str)
    counts = pd.crosstab(y_true, y_pred, rownames=[f"true {label_col}"], colnames=[f"pred {pred_col}"])
    labels = sorted(set(counts.index.astype(str)) | set(counts.columns.astype(str)))
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)
    row_norm = counts.astype(float).div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    counts_path = out_dir / "student_confusion_counts.csv"
    norm_path = out_dir / "student_confusion_row_normalized.csv"
    png_path = out_dir / "student_confusion_heatmap.png"
    counts.to_csv(counts_path)
    row_norm.to_csv(norm_path)

    plot_table = row_norm if normalize_rows else counts.astype(float)
    fig, ax = plt.subplots(
        figsize=(
            max(10, min(24, 0.42 * plot_table.shape[1] + 4)),
            max(9, min(24, 0.36 * plot_table.shape[0] + 3)),
        ),
        constrained_layout=True,
    )
    sns.heatmap(
        plot_table,
        cmap="Blues",
        vmin=0,
        vmax=float(plot_table.to_numpy().max()) if not normalize_rows else 1.0,
        ax=ax,
        cbar_kws={"label": "row-normalized fraction" if normalize_rows else "count"},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.35,
        linecolor="#d9e2ec",
    )
    acc = float(y_true.eq(y_pred).mean())
    title_prefix = f"{run_label} " if run_label else ""
    ax.set_title(f"{title_prefix}student confusion heatmap, accuracy={acc:.4f}")
    ax.set_xlabel(f"predicted: {pred_col}")
    ax.set_ylabel(f"true: {label_col}")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"counts": counts_path, "row_normalized": norm_path, "heatmap": png_path}


def write_classification_report(
    obs: pd.DataFrame,
    results_dir: str | Path,
    *,
    pred_col: str,
    label_col: str = "true_label",
    query_name: str = "query",
) -> pd.DataFrame:
    from sklearn.metrics import classification_report

    results_dir = Path(results_dir)
    mask = obs["ref_query_col"].astype(str).eq(query_name)
    report = pd.DataFrame(
        classification_report(
            obs.loc[mask, label_col].astype(str),
            obs.loc[mask, pred_col].astype(str),
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(results_dir / "classification_report_query.csv")
    return report


def branch_subset_metrics(
    obs: pd.DataFrame,
    prior_spec: Mapping[str, Any],
    *,
    pred_col: str,
    label_col: str = "true_label",
    query_name: str = "query",
) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, f1_score

    rows: list[dict[str, Any]] = []
    query_mask = obs["ref_query_col"].astype(str).eq(query_name)
    fine_classes = [str(x) for x in prior_spec.get("fine_classes", [])]
    for branch, desc_idx in prior_spec.get("branch_to_desc_indices", {}).items():
        labels = [fine_classes[int(i)] for i in desc_idx if int(i) < len(fine_classes)]
        mask = query_mask & obs[label_col].astype(str).isin(labels)
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "branch": str(branch),
                "n_query": int(mask.sum()),
                "accuracy": accuracy_score(obs.loc[mask, label_col].astype(str), obs.loc[mask, pred_col].astype(str)),
                "macro_f1": f1_score(
                    obs.loc[mask, label_col].astype(str),
                    obs.loc[mask, pred_col].astype(str),
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("branch", kind="mergesort") if rows else pd.DataFrame()


def write_teacher_stage_outputs(
    *,
    adata: ad.AnnData,
    out_dir: str | Path,
    stage: str,
    pred: pd.Series,
    soft: pd.DataFrame,
    latent: np.ndarray,
    prior_spec: Mapping[str, Any],
    reference_name: str,
    query_name: str,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_col = f"pred_{stage}"
    latent_key = f"X_{stage}"
    summary = add_prediction_outputs(
        adata,
        pred=pred,
        soft=soft,
        latent=latent,
        pred_col=pred_col,
        latent_key=latent_key,
        confidence_col=f"confidence_{stage}",
        entropy_col=f"entropy_{stage}",
        correct_col=f"correct_{stage}",
        reference_name=reference_name,
        query_name=query_name,
    )
    summary.to_frame().T.to_csv(out_dir / "summary_metrics.csv", index=False)
    counts = query_confusion_counts(adata.obs, pred_col, query_name=query_name)
    counts.to_csv(out_dir / "confusion_counts.csv")
    plot_confusion_heatmap(counts, title=f"{stage} query confusion", out_path=out_dir / "confusion_heatmap.png")
    write_classification_report(adata.obs, out_dir, pred_col=pred_col, query_name=query_name)
    branch_subset_metrics(adata.obs, prior_spec, pred_col=pred_col, query_name=query_name).to_csv(
        out_dir / "branch_subset_metrics.csv",
        index=False,
    )
    soft.to_csv(out_dir / "soft_probs_all_cells.csv")
    adata.write_h5ad(out_dir / "results.h5ad")
    write_json(out_dir / "config.json", {"stage": stage, "pred_col": pred_col, "latent_key": latent_key})
    return {
        "summary_csv": out_dir / "summary_metrics.csv",
        "confusion_counts_csv": out_dir / "confusion_counts.csv",
        "confusion_heatmap_png": out_dir / "confusion_heatmap.png",
        "results_h5ad": out_dir / "results.h5ad",
        "soft_probs_csv": out_dir / "soft_probs_all_cells.csv",
    }
