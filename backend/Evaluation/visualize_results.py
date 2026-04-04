"""
Load evaluation CSV rows and plot per-configuration / ablation comparisons.

Requires: pip install pandas matplotlib

Design:
- Per-prompt values are averaged when multiple rows share story_prompt before
  macro-averages across prompts (fair weight per prompt).
- Fix specialist_model_variant_tag when comparing boolean-flag toggles so
  AudioLDM2 specialist grid does not confound flag effects.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOL_FLAG_COLUMNS: Tuple[str, ...] = (
    "flag_fill_coverage_by_llm",
    "flag_use_dsp",
    "flag_use_movie_bgms",
    "flag_use_narrator",
    "flag_use_llm_to_predict_align",
    "flag_use_dsp_to_predict_align",
    "flag_use_avg_llm_and_dsp_to_predict_align",
    "flag_use_dl_based_llm_and_dsp_alignment_predictor",
)

DEFAULT_OBJECTIVE: Dict[str, Literal["maximize", "minimize"]] = {
    "clap_score": "maximize",
    "audio_richness_spectral_flatness": "maximize",
    "audio_richness_spectral_entropy": "maximize",
    "noise_floor_db": "maximize",
    "audio_onsets": "maximize",
    "yt_coverage_score": "maximize",
    "yt_sync_score": "maximize",
    "yt_coverage_and_sync_coverage_score": "maximize",
    "yt_coverage_and_sync_sync_score": "maximize",
    "cinematic_dynamic_range_db": "maximize",
    "cinematic_crest_factor": "maximize",
    "cinematic_spectral_flatness": "maximize",
    "cinematic_spectral_entropy": "maximize",
    "cinematic_spectral_centroid_hz": "maximize",
    "spectral_kl_divergence": "minimize",
    "fad_score": "minimize",
    "total_seconds": "minimize",
    "pipeline_total_seconds": "minimize",
    "evaluation_seconds": "minimize",
}


def _parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    up = s.astype(str).str.strip().str.upper()
    return up.map({"TRUE": True, "FALSE": False, "1": True, "0": False}).fillna(False)


def load_results_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    for col in BOOL_FLAG_COLUMNS:
        if col in df.columns:
            df[col] = _parse_bool_series(df[col])
    return df


def clean_results_df(
    df: pd.DataFrame,
    *,
    drop_empty_experiment_tag: bool = True,
    exclude_errors: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if drop_empty_experiment_tag and "experiment_tag" in out.columns:
        out = out[out["experiment_tag"].notna() & (out["experiment_tag"].astype(str).str.strip() != "")]
    if exclude_errors and "error" in out.columns:
        err = out["error"]
        mask = err.isna() | (err.astype(str).str.strip() == "")
        out = out[mask]
    return out


def infer_baseline_config(df: pd.DataFrame) -> Dict[str, Any]:
    """First row with baseline experiment and baseline_tango2 specialist."""
    sub = df[
        (df["experiment_tag"].astype(str) == "baseline")
        & (df["specialist_model_variant_tag"].astype(str) == "baseline_tango2")
    ]
    if sub.empty:
        sub = df[df["experiment_tag"].astype(str) == "baseline"]
    if sub.empty:
        raise ValueError("No baseline rows found to infer default configuration.")
    row = sub.iloc[0]
    cfg: Dict[str, Any] = {}
    for c in BOOL_FLAG_COLUMNS:
        if c in row.index:
            cfg[c] = bool(row[c])
    for c in ("decide_audio_model_name", "specialist_model_variant_tag"):
        if c in row.index and pd.notna(row[c]):
            cfg[c] = str(row[c])
    return cfg


def filter_by_configuration(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    out = df
    for key, val in config.items():
        if key not in out.columns:
            raise KeyError(f"Unknown column in config: {key}")
        if key in BOOL_FLAG_COLUMNS:
            v = bool(val)
            out = out[out[key] == v]
        else:
            out = out[out[key].astype(str) == str(val)]
    return out


def resolve_objective(metric_col: str, objective: Optional[Literal["maximize", "minimize"]]) -> Literal["maximize", "minimize"]:
    if objective is not None:
        return objective
    if metric_col in DEFAULT_OBJECTIVE:
        return DEFAULT_OBJECTIVE[metric_col]
    return "maximize"


@dataclass
class ConfigurationPlotInsights:
    metric_col: str
    n_prompts: int
    mean_over_prompts: float
    best_prompt: str
    best_value: float
    worst_prompt: str
    worst_value: float
    objective: str


def plot_metric_for_configuration(
    df: pd.DataFrame,
    metric_col: str,
    configuration: Optional[Mapping[str, Any]] = None,
    *,
    output_path: str,
    aggregate_how: Literal["mean"] = "mean",
    objective: Optional[Literal["maximize", "minimize"]] = None,
    exclude_errors: bool = True,
    prompt_label_max_len: int = 56,
    dpi: int = 150,
    save_pdf: bool = False,
) -> Tuple[pd.DataFrame, ConfigurationPlotInsights]:
    """
    Filter rows to ``configuration`` (defaults merged from infer_baseline_config),
    aggregate metric per story_prompt, plot horizontal bars, save figure.
    Returns (per_prompt DataFrame, insights).
    """
    obj = resolve_objective(metric_col, objective)
    work = clean_results_df(df, exclude_errors=exclude_errors) if exclude_errors else df.copy()

    cfg: Dict[str, Any] = dict(infer_baseline_config(work))
    if configuration:
        cfg.update(dict(configuration))

    filt = filter_by_configuration(work, cfg)
    if metric_col not in filt.columns:
        raise KeyError(f"Unknown metric column: {metric_col}")

    filt = filt.assign(**{metric_col: pd.to_numeric(filt[metric_col], errors="coerce")})
    filt = filt[filt[metric_col].notna()]
    if filt.empty:
        raise ValueError("No rows left after filtering for configuration and valid metric.")

    if aggregate_how == "mean":
        g = filt.groupby("story_prompt", as_index=False)[metric_col].mean()
    else:
        raise ValueError(f"Unsupported aggregate_how: {aggregate_how}")

    g = g.sort_values(metric_col, ascending=(obj == "minimize"))
    prompts = g["story_prompt"].astype(str).tolist()
    values = g[metric_col].astype(float).tolist()

    def short_label(p: str, i: int) -> str:
        s = p if len(p) <= prompt_label_max_len else p[: prompt_label_max_len - 1] + "…"
        return f"{i + 1}. {s}"

    labels = [short_label(p, i) for i, p in enumerate(prompts)]
    fig_h = max(6.0, 0.22 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(labels)), values, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(metric_col)
    ax.set_title(f"{metric_col} per story_prompt\nconfig={cfg}")
    ax.invert_yaxis()
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(output_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(os.path.splitext(output_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)

    mean_v = float(sum(values) / len(values)) if values else float("nan")
    if obj == "maximize":
        bi = max(range(len(values)), key=lambda i: values[i])
        wi = min(range(len(values)), key=lambda i: values[i])
    else:
        bi = min(range(len(values)), key=lambda i: values[i])
        wi = max(range(len(values)), key=lambda i: values[i])

    insights = ConfigurationPlotInsights(
        metric_col=metric_col,
        n_prompts=len(values),
        mean_over_prompts=mean_v,
        best_prompt=prompts[bi],
        best_value=float(values[bi]),
        worst_prompt=prompts[wi],
        worst_value=float(values[wi]),
        objective=obj,
    )
    print(
        f"[{metric_col}] mean_over_prompts={insights.mean_over_prompts:.6g} "
        f"best_prompt (truncated)={insights.best_prompt[:prompt_label_max_len]!r} "
        f"best_value={insights.best_value:.6g} objective={obj}"
    )
    return g.rename(columns={metric_col: f"mean_{metric_col}"}), insights


def _per_prompt_means_for_slice(
    slice_df: pd.DataFrame,
    metric_col: str,
) -> pd.Series:
    s = pd.to_numeric(slice_df[metric_col], errors="coerce")
    tmp = slice_df.assign(_m=s)
    tmp = tmp[tmp["_m"].notna()]
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp.groupby("story_prompt")["_m"].mean()


def _ordered_experiment_tags(tags: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)

    def sort_key(t: str) -> Tuple[int, str]:
        if t == "baseline":
            return (0, t)
        if t.startswith("decide_model_"):
            return (1, t)
        if t.startswith("toggle_"):
            return (2, t)
        return (3, t)

    return sorted(ordered, key=sort_key)


@dataclass
class AblationRow:
    label: str
    mean_over_prompts: float
    best_single_prompt_value: float
    n_prompts: int


def _sanitize_metric_for_path(metric_col: str) -> str:
    return metric_col.replace(os.sep, "_").replace("/", "_")


def _iter_ablation_per_prompt_series(
    work: pd.DataFrame,
    metric_col: str,
    *,
    specialist_model_variant_tag: str,
    include_decide_model: bool,
    include_specialist_ablations: bool,
) -> Iterable[Tuple[str, pd.Series]]:
    """
    Yields (display_label, Series indexed by story_prompt with mean metric per prompt)
    in the same order as compare_ablations_for_metric.
    """
    tag_series = work["experiment_tag"].astype(str)
    all_tags = _ordered_experiment_tags(tag_series.unique())

    def one(ex_tag: str, label: Optional[str] = None) -> Optional[Tuple[str, pd.Series]]:
        lab = label or ex_tag
        sub = work[
            (tag_series == ex_tag) & (work["specialist_model_variant_tag"].astype(str) == specialist_model_variant_tag)
        ]
        pm = _per_prompt_means_for_slice(sub, metric_col)
        if pm.empty:
            return None
        return (lab, pm)

    for ex_tag in all_tags:
        if ex_tag == "baseline":
            out = one("baseline")
            if out:
                yield out
            continue
        if ex_tag.startswith("decide_model_"):
            if include_decide_model:
                out = one(ex_tag)
                if out:
                    yield out
            continue
        if ex_tag.startswith("toggle_"):
            out = one(ex_tag)
            if out:
                yield out
            continue
        out = one(ex_tag)
        if out:
            yield out

    if include_specialist_ablations:
        baseline_only = work[tag_series == "baseline"]
        spec_tags = sorted(
            baseline_only["specialist_model_variant_tag"].dropna().astype(str).unique(),
            key=lambda x: (0 if x == "baseline_tango2" else 1, x),
        )
        for st in spec_tags:
            if st == specialist_model_variant_tag:
                continue
            sub = baseline_only[baseline_only["specialist_model_variant_tag"].astype(str) == st]
            pm = _per_prompt_means_for_slice(sub, metric_col)
            lab = f"baseline | specialist={st}"
            if not pm.empty:
                yield (lab, pm)


def ablation_per_prompt_matrix(
    df: pd.DataFrame,
    metric_col: str,
    *,
    objective: Optional[Literal["maximize", "minimize"]] = None,
    specialist_model_variant_tag: str = "baseline_tango2",
    include_decide_model: bool = True,
    include_specialist_ablations: bool = False,
    exclude_errors: bool = True,
) -> pd.DataFrame:
    """
    Build a matrix: rows = story_prompt, columns = ablation labels, values = mean metric.
    Keeps only prompts that have a finite value for every ablation column (intersection).
    """
    resolve_objective(metric_col, objective)
    work = clean_results_df(df, exclude_errors=exclude_errors) if exclude_errors else df.copy()
    if metric_col not in work.columns:
        raise KeyError(f"Unknown metric column: {metric_col}")

    pieces: Dict[str, pd.Series] = {}
    for lab, ser in _iter_ablation_per_prompt_series(
        work,
        metric_col,
        specialist_model_variant_tag=specialist_model_variant_tag,
        include_decide_model=include_decide_model,
        include_specialist_ablations=include_specialist_ablations,
    ):
        if lab in pieces:
            continue
        pieces[lab] = ser

    if not pieces:
        return pd.DataFrame()

    mat = pd.concat(pieces, axis=1)
    mat.columns = [str(c) for c in mat.columns]
    return mat.dropna(how="any")


def summarize_prompt_wins(
    matrix: pd.DataFrame,
    objective: Literal["maximize", "minimize"],
    *,
    baseline_label: str = "baseline",
) -> pd.DataFrame:
    """
    Per ablation: strict wins (sole best on a prompt), fractional wins (1/k on k-way ties),
    win rates, and beats_baseline count if baseline column exists.
    Ties: each tied ablation receives 1/k of a win for that prompt.
    """
    if matrix.empty or matrix.shape[1] < 1:
        return pd.DataFrame()

    ascending = objective == "minimize"
    n = len(matrix)

    strict = {c: 0 for c in matrix.columns}
    fractional = {c: 0.0 for c in matrix.columns}

    for _, row in matrix.iterrows():
        vals = row.to_numpy(dtype=float)
        if ascending:
            best = np.nanmin(vals)
            is_best = np.isclose(vals, best, rtol=1e-9, atol=1e-12)
        else:
            best = np.nanmax(vals)
            is_best = np.isclose(vals, best, rtol=1e-9, atol=1e-12)
        k = int(is_best.sum())
        if k == 0:
            continue
        share = 1.0 / k
        for j, col in enumerate(matrix.columns):
            if is_best[j]:
                fractional[col] += share
                if k == 1:
                    strict[col] += 1

    rows = []
    for col in matrix.columns:
        rows.append(
            {
                "ablation": col,
                "prompts_won_strict": strict[col],
                "prompts_won_fractional": fractional[col],
                "win_rate_strict": strict[col] / n if n else float("nan"),
                "win_rate_fractional": fractional[col] / n if n else float("nan"),
            }
        )
    out = pd.DataFrame(rows)

    if baseline_label in matrix.columns:
        b = matrix[baseline_label].to_numpy(dtype=float)
        beats = []
        for col in matrix.columns:
            if col == baseline_label:
                beats.append(float("nan"))
                continue
            v = matrix[col].to_numpy(dtype=float)
            if ascending:
                beats.append(float(np.sum(v < b)))
            else:
                beats.append(float(np.sum(v > b)))
        out["beats_baseline_count"] = beats
        out["beats_baseline_rate"] = out["beats_baseline_count"] / n if n else float("nan")

    return out.sort_values("win_rate_fractional", ascending=False, ignore_index=True)


def mean_rank_per_ablation(
    matrix: pd.DataFrame,
    objective: Literal["maximize", "minimize"],
) -> pd.DataFrame:
    """Mean rank per column (1 = best). Ties get the average rank."""
    if matrix.empty:
        return pd.DataFrame()
    ascending = objective == "minimize"
    r = matrix.rank(axis=1, ascending=ascending, method="average")
    out = r.mean(axis=0).reset_index()
    out.columns = ["ablation", "mean_rank"]
    return out.sort_values("mean_rank", ascending=True, ignore_index=True)


def plot_win_count_bars(
    wins_summary: pd.DataFrame,
    *,
    output_path: str,
    metric_col: str,
    dpi: int = 150,
) -> None:
    """Horizontal bars: fractional win share (best for most prompts, ties split)."""
    if wins_summary.empty or "win_rate_fractional" not in wins_summary.columns:
        return
    sub = wins_summary.sort_values("win_rate_fractional", ascending=True).copy()
    labels = sub["ablation"].astype(str).tolist()
    vals = sub["win_rate_fractional"].tolist()
    fig_h = max(5.0, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(labels)), vals, color="teal")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Win rate (fractional; ties split 1/k)")
    ax.set_title(f"Per-prompt winners — {metric_col}")
    ax.set_xlim(0, max(0.05, max(vals) * 1.15) if vals else 1.0)
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(output_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmap(
    matrix: pd.DataFrame,
    *,
    output_path: str,
    metric_col: str,
    zscore_rows: bool = True,
    max_y_labels: int = 40,
    dpi: int = 150,
) -> None:
    """Heatmap prompts × ablations (matplotlib only). Optional per-row z-score."""
    if matrix.empty:
        return
    Z = matrix.to_numpy(dtype=float)
    if zscore_rows:
        m = Z.mean(axis=1, keepdims=True)
        s = Z.std(axis=1, keepdims=True)
        s = np.where(s < 1e-12, 1.0, s)
        Z = (Z - m) / s

    n_prompts, n_ab = Z.shape
    fig_h = min(14, max(5, 0.16 * n_prompts))
    fig_w = max(8, 0.45 * n_ab)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks(range(n_ab))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=55, ha="right", fontsize=7)
    step = max(1, n_prompts // max_y_labels)
    yticks = list(range(0, n_prompts, step))
    ax.set_yticks(yticks)
    prompts = matrix.index.astype(str).tolist()
    ax.set_yticklabels([prompts[i][:40] + "…" if len(prompts[i]) > 40 else prompts[i] for i in yticks], fontsize=6)
    ttl = f"{'Row z-scored ' if zscore_rows else ''}{metric_col}: prompts × ablations"
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(output_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_margin_vs_baseline(
    matrix: pd.DataFrame,
    *,
    output_path: str,
    metric_col: str,
    objective: Literal["maximize", "minimize"],
    baseline_label: str = "baseline",
    dpi: int = 150,
) -> None:
    """Mean score minus baseline per ablation (intersection prompts only)."""
    if matrix.empty or baseline_label not in matrix.columns:
        return
    b = matrix[baseline_label].astype(float)
    means = []
    labels = []
    for col in matrix.columns:
        delta = (matrix[col].astype(float) - b).mean()
        means.append(delta)
        labels.append(col)
    sub = pd.DataFrame({"ablation": labels, "mean_margin_vs_baseline": means})
    sub = sub.sort_values("mean_margin_vs_baseline", ascending=(objective == "minimize"))
    fig_w = max(9.0, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    colors = ["gray" if lab == baseline_label else "steelblue" for lab in sub["ablation"]]
    x = range(len(sub))
    ax.bar(x, sub["mean_margin_vs_baseline"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub["ablation"].tolist(), rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(f"Mean ({metric_col} - {baseline_label})")
    ax.set_title(f"Average margin vs {baseline_label} — {metric_col}")
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(output_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compare_ablations_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    *,
    objective: Optional[Literal["maximize", "minimize"]] = None,
    specialist_model_variant_tag: str = "baseline_tango2",
    output_dir: str,
    include_decide_model: bool = True,
    include_specialist_ablations: bool = False,
    exclude_errors: bool = True,
    dpi: int = 150,
) -> Tuple[pd.DataFrame, str]:
    """
    For each experiment_tag (optionally plus specialist-only bars), compute macro mean
    across prompts and best single-prompt mean; save two bar plots and a text report.
    Returns (summary DataFrame, report text).
    """
    obj = resolve_objective(metric_col, objective)
    work = clean_results_df(df, exclude_errors=exclude_errors) if exclude_errors else df.copy()
    if metric_col not in work.columns:
        raise KeyError(f"Unknown metric column: {metric_col}")

    os.makedirs(output_dir, exist_ok=True)

    rows_out: List[AblationRow] = []

    tag_series = work["experiment_tag"].astype(str)
    all_tags = _ordered_experiment_tags(tag_series.unique())

    def process_tag(ex_tag: str, label: Optional[str] = None) -> None:
        lab = label or ex_tag
        sub = work[
            (tag_series == ex_tag) & (work["specialist_model_variant_tag"].astype(str) == specialist_model_variant_tag)
        ]
        pm = _per_prompt_means_for_slice(sub, metric_col)
        if pm.empty:
            rows_out.append(
                AblationRow(label=lab, mean_over_prompts=float("nan"), best_single_prompt_value=float("nan"), n_prompts=0)
            )
            return
        mean_v = float(pm.mean())
        if obj == "maximize":
            best_v = float(pm.max())
        else:
            best_v = float(pm.min())
        rows_out.append(
            AblationRow(label=lab, mean_over_prompts=mean_v, best_single_prompt_value=best_v, n_prompts=int(pm.shape[0]))
        )

    for ex_tag in all_tags:
        if ex_tag == "baseline":
            process_tag("baseline")
            continue
        if ex_tag.startswith("decide_model_"):
            if include_decide_model:
                process_tag(ex_tag)
            continue
        if ex_tag.startswith("toggle_"):
            process_tag(ex_tag)
            continue
        process_tag(ex_tag)

    if include_specialist_ablations:
        baseline_only = work[tag_series == "baseline"]
        spec_tags = sorted(
            baseline_only["specialist_model_variant_tag"].dropna().astype(str).unique(),
            key=lambda x: (0 if x == "baseline_tango2" else 1, x),
        )
        for st in spec_tags:
            if st == specialist_model_variant_tag:
                continue
            sub = baseline_only[baseline_only["specialist_model_variant_tag"].astype(str) == st]
            pm = _per_prompt_means_for_slice(sub, metric_col)
            lab = f"baseline | specialist={st}"
            if pm.empty:
                rows_out.append(
                    AblationRow(label=lab, mean_over_prompts=float("nan"), best_single_prompt_value=float("nan"), n_prompts=0)
                )
            else:
                mean_v = float(pm.mean())
                best_v = float(pm.max()) if obj == "maximize" else float(pm.min())
                rows_out.append(
                    AblationRow(
                        label=lab, mean_over_prompts=mean_v, best_single_prompt_value=best_v, n_prompts=int(pm.shape[0])
                    )
                )

    summary = pd.DataFrame([r.__dict__ for r in rows_out])

    def _plot_bars(column: str, title: str, fname: str) -> None:
        sub2 = summary.dropna(subset=[column]).copy()
        if sub2.empty:
            return
        sub2 = sub2.sort_values(column, ascending=(obj == "minimize"))
        labels = sub2["label"].tolist()
        vals = sub2[column].tolist()
        fig_w = max(10.0, 0.35 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        color = "seagreen" if column == "mean_over_prompts" else "darkorange"
        ax.bar(range(len(labels)), vals, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel(column)
        ax.set_title(title)
        fig.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    _plot_bars("mean_over_prompts", f"Mean over prompts — {metric_col}", f"ablations_mean_{metric_col}.png")
    _plot_bars("best_single_prompt_value", f"Best single-prompt mean — {metric_col}", f"ablations_best_{metric_col}.png")

    valid_mean = summary.dropna(subset=["mean_over_prompts"])
    valid_best = summary.dropna(subset=["best_single_prompt_value"])

    report_lines: List[str] = [
        f"# Ablation comparison: {metric_col}",
        f"objective: {obj}",
        f"specialist_model_variant_tag filter: {specialist_model_variant_tag!r}",
        "",
    ]

    if not valid_mean.empty:
        if obj == "maximize":
            win_mean = valid_mean.loc[valid_mean["mean_over_prompts"].idxmax()]
        else:
            win_mean = valid_mean.loc[valid_mean["mean_over_prompts"].idxmin()]
        report_lines.append(f"## Best configuration by mean over prompts: {win_mean['label']}")
        report_lines.append(f"value: {win_mean['mean_over_prompts']:.6g} (n_prompts={win_mean['n_prompts']})")
        report_lines.append("")
    if not valid_best.empty:
        if obj == "maximize":
            win_best = valid_best.loc[valid_best["best_single_prompt_value"].idxmax()]
        else:
            win_best = valid_best.loc[valid_best["best_single_prompt_value"].idxmin()]
        report_lines.append(f"## Best configuration by peak single-prompt score: {win_best['label']}")
        report_lines.append(f"value: {win_best['best_single_prompt_value']:.6g} (n_prompts={win_best['n_prompts']})")
        report_lines.append("")

    report_lines.append("## Table")
    report_lines.append(summary.to_string(index=False))

    if not valid_mean.empty:
        baseline_row = valid_mean[valid_mean["label"] == "baseline"]
        if not baseline_row.empty:
            b = float(baseline_row.iloc[0]["mean_over_prompts"])
            others = valid_mean[valid_mean["label"] != "baseline"].copy()
            if not others.empty:
                others = others.copy()
                others["delta_vs_baseline"] = others["mean_over_prompts"] - b
                if obj == "maximize":
                    worst = others.loc[others["delta_vs_baseline"].idxmin()]
                    report_lines.append(
                        f"\n## Largest mean drop vs baseline\n{worst['label']} (delta={worst['delta_vs_baseline']:.6g})"
                    )
                else:
                    worst = others.loc[others["delta_vs_baseline"].idxmax()]
                    report_lines.append(
                        f"\n## Largest mean increase vs baseline (worse)\n{worst['label']} (delta={worst['delta_vs_baseline']:.6g})"
                    )

    report_text = "\n".join(report_lines)
    rep_path = os.path.join(output_dir, f"ablation_report_{metric_col}.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote report: {rep_path}")

    return summary, report_text


def compare_ablations_for_metric_extended(
    df: pd.DataFrame,
    metric_col: str,
    *,
    objective: Optional[Literal["maximize", "minimize"]] = None,
    specialist_model_variant_tag: str = "baseline_tango2",
    output_dir: str,
    include_decide_model: bool = True,
    include_specialist_ablations: bool = False,
    exclude_errors: bool = True,
    dpi: int = 150,
    zscore_heatmap_rows: bool = True,
) -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Writes outputs under ``output_dir / <metric_sanitized> /``:
    original ablation bar charts + report, plus per-prompt matrix CSV, win summary CSV,
    mean-rank table, consensus plots (heatmap, win-rate bars, margin vs baseline).
    Returns (summary, report_text, matrix, wins_summary, mean_ranks).
    """
    obj = resolve_objective(metric_col, objective)
    run_dir = os.path.join(output_dir, _sanitize_metric_for_path(metric_col))
    os.makedirs(run_dir, exist_ok=True)

    summary, report_text = compare_ablations_for_metric(
        df,
        metric_col,
        objective=objective,
        specialist_model_variant_tag=specialist_model_variant_tag,
        output_dir=run_dir,
        include_decide_model=include_decide_model,
        include_specialist_ablations=include_specialist_ablations,
        exclude_errors=exclude_errors,
        dpi=dpi,
    )

    matrix = ablation_per_prompt_matrix(
        df,
        metric_col,
        objective=objective,
        specialist_model_variant_tag=specialist_model_variant_tag,
        include_decide_model=include_decide_model,
        include_specialist_ablations=include_specialist_ablations,
        exclude_errors=exclude_errors,
    )

    if matrix.empty or matrix.shape[1] < 2:
        print("[consensus] Skipping per-prompt consensus: insufficient ablation columns or no intersection prompts.")
        return summary, report_text, matrix, pd.DataFrame(), pd.DataFrame()

    wins = summarize_prompt_wins(matrix, obj)
    ranks = mean_rank_per_ablation(matrix, obj)

    matrix_path = os.path.join(run_dir, f"ablation_matrix_{metric_col}.csv")
    matrix.to_csv(matrix_path, encoding="utf-8")
    wins_path = os.path.join(run_dir, f"win_summary_{metric_col}.csv")
    wins.to_csv(wins_path, index=False, encoding="utf-8")
    ranks_path = os.path.join(run_dir, f"mean_ranks_{metric_col}.csv")
    ranks.to_csv(ranks_path, index=False, encoding="utf-8")

    plot_win_count_bars(
        wins,
        output_path=os.path.join(run_dir, f"ablations_win_rate_{metric_col}.png"),
        metric_col=metric_col,
        dpi=dpi,
    )
    plot_ablation_heatmap(
        matrix,
        output_path=os.path.join(run_dir, f"ablations_heatmap_{metric_col}.png"),
        metric_col=metric_col,
        zscore_rows=zscore_heatmap_rows,
        dpi=dpi,
    )
    plot_margin_vs_baseline(
        matrix,
        output_path=os.path.join(run_dir, f"ablations_margin_vs_baseline_{metric_col}.png"),
        metric_col=metric_col,
        objective=obj,
        dpi=dpi,
    )

    extra = [
        "",
        "## Per-prompt consensus (intersection only)",
        f"prompts_in_intersection: {len(matrix)}",
        f"ablations: {matrix.shape[1]}",
        "",
        "### Win summary (fractional wins split on ties)",
        wins.to_string(index=False),
        "",
        "### Mean rank (1 = best)",
        ranks.to_string(index=False),
        "",
        f"Saved: {matrix_path}",
        f"Saved: {wins_path}",
        f"Saved: {ranks_path}",
    ]
    report_path = os.path.join(run_dir, f"ablation_report_{metric_col}.txt")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(extra))
    print(f"Appended consensus section to {report_path}")

    return summary, report_text + "\n" + "\n".join(extra), matrix, wins, ranks


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize evaluation CSV results.")
    p.add_argument("--csv", type=str, default="", help="Path to final_results.csv")
    p.add_argument("--metric", type=str, default="clap_score")
    p.add_argument("--outdir", type=str, default="", help="Output directory for plots and reports")
    p.add_argument(
        "--specialist-tag",
        type=str,
        default="baseline_tango2",
        help="specialist_model_variant_tag for ablation slices",
    )
    p.add_argument("--no-decide-model", action="store_true", help="Exclude decide_model_* from ablation bars")
    p.add_argument(
        "--specialist-ablations",
        action="store_true",
        help="Add baseline experiment rows for each specialist variant",
    )
    p.add_argument("--objective", type=str, default="", choices=["", "maximize", "minimize"])
    p.add_argument(
        "--skip-per-prompt-plot",
        action="store_true",
        help="Do not save baseline-configuration per-prompt bar chart",
    )
    p.add_argument(
        "--no-consensus",
        action="store_true",
        help="Skip per-prompt matrix, win rates, heatmap (only mean/best ablation bars)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv or os.path.join(here, "Results", "final_results.csv")
    outdir = args.outdir or os.path.join(here, "Results", "plots")
    obj: Optional[Literal["maximize", "minimize"]] = None
    if args.objective:
        obj = args.objective  # type: ignore[assignment]

    df = load_results_csv(csv_path)
    df_clean = clean_results_df(df)

    if args.no_consensus:
        compare_ablations_for_metric(
            df_clean,
            args.metric,
            objective=obj,
            specialist_model_variant_tag=args.specialist_tag,
            output_dir=outdir,
            include_decide_model=not args.no_decide_model,
            include_specialist_ablations=args.specialist_ablations,
        )
        cfg_base = outdir
    else:
        compare_ablations_for_metric_extended(
            df_clean,
            args.metric,
            objective=obj,
            specialist_model_variant_tag=args.specialist_tag,
            output_dir=outdir,
            include_decide_model=not args.no_decide_model,
            include_specialist_ablations=args.specialist_ablations,
        )
        cfg_base = os.path.join(outdir, _sanitize_metric_for_path(args.metric))

    if not args.skip_per_prompt_plot:
        cfg_plot_path = os.path.join(cfg_base, f"per_prompt_{args.metric}_baseline.png")
        plot_metric_for_configuration(
            df_clean,
            args.metric,
            configuration=None,
            output_path=cfg_plot_path,
            objective=obj,
        )


if __name__ == "__main__":
    main()
