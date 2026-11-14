import numpy as np
import pandas as pd
import plotly.express as px

PURPLE_SCALE = ["#efe5ff", "#d8c1ff", "#b38af0", "#8d58d8", "#6732a6", "#3f176d"]


def _style(fig, height=340):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        font=dict(family="Inter, Arial, sans-serif", color="#352246"),
        title=dict(font=dict(size=15, color="#35145f"), x=0.02),
        margin=dict(l=12, r=12, t=42, b=36),
        legend=dict(bgcolor="rgba(255,255,255,0.72)", bordercolor="rgba(103,50,166,.10)", borderwidth=1, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#2d1746", font_color="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(103,50,166,.15)")
    fig.update_yaxes(gridcolor="rgba(103,50,166,.10)", zeroline=False)
    return fig


def _is_id_like(series, name):
    name_l = str(name).lower()
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return ("id" in name_l and unique_ratio > 0.5) or unique_ratio > 0.98


def _useful_numeric(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    return [c for c in numeric if not _is_id_like(df[c], c)]


def _useful_categorical(df):
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()
    return [c for c in categorical if df[c].nunique(dropna=True) <= 20]


def analyze_dataframe(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()
    useful_numeric = _useful_numeric(df)
    missing = df.isna().sum().sort_values(ascending=False)
    highlights = [
        f"Dataset contains {len(df):,} rows and {len(df.columns)} columns.",
        f"{len(numeric)} numeric and {len(categorical)} non-numeric columns detected.",
        f"Total missing cells: {int(df.isna().sum().sum()):,}.",
        f"Duplicate rows: {int(df.duplicated().sum()):,}.",
    ]
    if useful_numeric:
        var = df[useful_numeric].var().sort_values(ascending=False)
        if len(var):
            highlights.append(f"Highest-variance business metric: {var.index[0]}.")
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    highest_variance_metric = None
    if useful_numeric:
        var = df[useful_numeric].var().sort_values(ascending=False)
        if len(var):
            highest_variance_metric = var.index[0]

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_count": int(len(numeric)),
        "categorical_count": int(len(categorical)),
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "missing_cells": missing_cells,
        "duplicates": duplicates,
        "highest_variance_metric": highest_variance_metric,
        "missing_df": missing[missing > 0].rename("missing").to_frame(),
        "highlights": highlights,
    }


def build_eda_figures(df):
    figs = []
    numeric = _useful_numeric(df)
    categorical = _useful_categorical(df)
    preferred = ["Sales", "Profit", "Price", "Quantity_Sold", "Customer_Rating", "Reviews"]
    metric = next((c for c in preferred if c in numeric), numeric[0] if numeric else None)

    if metric:
        fig = px.histogram(df, x=metric, nbins=20, title=f"Distribution of {metric}", color_discrete_sequence=["#7a4bc1"])
        fig.update_traces(marker_line_width=0.45, marker_line_color="#4b2378")
        fig.update_layout(bargap=0.08)
        figs.append((f"Distribution — {metric}", _style(fig, 330)))

    if len(numeric) >= 2:
        corr = df[numeric].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=PURPLE_SCALE,
                        title="Correlation Heatmap — Business Metrics")
        figs.append(("Correlation Heatmap — Business Metrics", _style(fig, 380)))

    if categorical and metric:
        cat = next((c for c in ["Category", "Color", "Style", "Month"] if c in categorical), categorical[0])
        g = (df.groupby(cat, dropna=False)[metric].mean().reset_index().sort_values(metric, ascending=False))
        fig = px.bar(g, x=cat, y=metric, title=f"Average {metric} by {cat}",
                     color=metric, color_continuous_scale=PURPLE_SCALE)
        fig.update_layout(bargap=0.32)
        figs.append((f"Average {metric} by {cat}", _style(fig, 330)))
    return figs
