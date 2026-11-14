import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor


def _style(fig, height=330):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,.65)",
        font=dict(color="#352246"),
        title=dict(font=dict(size=15, color="#35145f"), x=.02),
        margin=dict(l=12, r=12, t=42, b=34),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(103,50,166,.10)", zeroline=False)
    return fig


def infer_task(y, requested):
    if requested != "Auto":
        return requested
    return "Classification" if y.dtype == "object" or y.nunique(dropna=True) <= 10 else "Regression"


def _preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns
    categorical = X.select_dtypes(exclude=np.number).columns

    transformers = []
    if len(numeric):
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if len(categorical):
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers)


def _looks_like_identifier(series, column_name, n_rows):
    """Return True for columns that are identifiers rather than useful predictors."""
    name = str(column_name).strip().lower()
    id_keywords = (
        "_id", "id_", "identifier", "product_id", "customer_id",
        "order_id", "transaction_id", "sku", "image_file", "image_name",
        "product_name", "customer_name", "order_name"
    )

    if name in {"id", "productid", "customerid", "orderid", "sku"}:
        return True
    if any(k in name for k in id_keywords):
        return True

    # A string column with almost every row unique is normally an identifier.
    if n_rows > 0 and series.nunique(dropna=True) >= max(20, int(0.90 * n_rows)):
        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            return True

    # Numeric columns with almost every value unique are also usually IDs.
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) >= max(20, int(0.98 * n_rows)):
        if name.endswith("id") or name.startswith("id"):
            return True

    return False


def _detect_leakage_columns(X, target):
    """Detect obvious identifier and outcome-derived leakage for business datasets."""
    excluded = []
    target_name = str(target).strip().lower()

    for col in X.columns:
        name = str(col).strip().lower()

        # Always remove identifiers from predictive modelling.
        if _looks_like_identifier(X[col], col, len(X)):
            excluded.append((col, "identifier / label-like field"))
            continue

        # These are common post-outcome/business-derived fields. If one is the
        # target, its paired outcome should not be used as a predictor.
        outcome_pairs = {
            "profit": {"sales", "revenue"},
            "sales": {"profit", "revenue"},
            "revenue": {"profit", "sales"},
            "quantity_sold": {"sales", "revenue"},
        }
        if target_name in outcome_pairs and name in outcome_pairs[target_name]:
            excluded.append((col, "possible outcome leakage"))

    return excluded


def train_models(df, target, task):
    data = df.dropna(subset=[target]).copy()
    X_raw = data.drop(columns=[target])
    y = data[target]

    # Remove identifiers and obvious post-outcome leakage before the train/test split.
    excluded = _detect_leakage_columns(X_raw, target)
    excluded_names = [name for name, _ in excluded]
    X = X_raw.drop(columns=excluded_names, errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("No valid predictor columns remain after removing identifiers/leakage fields.")

    if task == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=.08,
                subsample=.9,
                colsample_bytree=.9,
                random_state=42,
                n_jobs=-1,
            ),
        }
    else:
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=.08,
                subsample=.9,
                colsample_bytree=.9,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
        }

    stratify = None
    if task == "Classification":
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=stratify
    )

    rows, fitted = [], {}

    for name, model in models.items():
        pipe = Pipeline([("prep", _preprocessor(X)), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

        if task == "Regression":
            rows.append(
                {
                    "Model": name,
                    "R2": r2_score(yte, pred),
                    "RMSE": mean_squared_error(yte, pred) ** .5,
                    "MAE": mean_absolute_error(yte, pred),
                }
            )
        else:
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(yte, pred),
                    "F1": f1_score(yte, pred, average="weighted"),
                }
            )
        fitted[name] = pipe

    metrics = pd.DataFrame(rows)
    score_col = "R2" if task == "Regression" else "Accuracy"
    best_name = metrics.sort_values(score_col, ascending=False).iloc[0]["Model"]
    best = fitted[best_name]

    # IMPORTANT FIX:
    # Use permutation importance on the held-out raw features instead of
    # one-hot coefficients/tree importances. This gives one importance score
    # per original business feature and prevents Product_Name/Image_File
    # categories from dominating the chart.
    try:
        scoring = "r2" if task == "Regression" else "accuracy"
        perm = permutation_importance(
            best,
            Xte,
            yte,
            scoring=scoring,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        fi = pd.DataFrame(
            {
                "feature": Xte.columns,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }
        )
        fi = fi.replace([np.inf, -np.inf], np.nan).dropna(subset=["importance"])
        fi = fi.sort_values("importance", ascending=False).head(12)

        importance_fig = px.bar(
            fi.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Permutation Feature Importance",
            color="importance",
            color_continuous_scale=["#d8c1ff", "#8f5de7", "#4b1f78"],
            hover_data={"importance": ":.4f", "feature": True},
        )
        importance_fig.update_yaxes(title=None)
        importance_fig.update_xaxes(title="Mean score decrease")
        importance_fig.update_layout(bargap=0.24, coloraxis_showscale=False)
        importance_fig = _style(importance_fig, max(360, 45 + 30 * len(fi)))

        # Make negative permutation scores visible but explainable.
        importance_note = (
            "Permutation importance measures how much the held-out model score "
            "changes when each original feature is shuffled. Values near zero "
            "indicate little predictive contribution."
        )
    except Exception:
        importance_fig = _style(px.bar(title="Feature importance unavailable"))
        importance_fig.add_annotation(
            text="Importance could not be calculated for the selected model.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#5e5270", size=13),
        )
        importance_note = "Feature importance could not be calculated."

    comparison_fig = px.bar(
        metrics,
        x="Model",
        y=score_col,
        title=f"Model comparison ({score_col})",
        color=score_col,
        color_continuous_scale=["#d8c1ff", "#8f5de7", "#4b1f78"],
    )
    comparison_fig.update_layout(bargap=0.35)
    comparison_fig = _style(comparison_fig)

    best_score = metrics.loc[metrics.Model == best_name, score_col].iloc[0]
    return {
        "task": task,
        "metrics": metrics.round(4),
        "comparison_fig": comparison_fig,
        "importance_fig": importance_fig,
        "best_model_text": f"Best model: {best_name} ({score_col}={best_score:.4f})",
        "model": best,
        "excluded_features": excluded,
        "predictor_columns": list(X.columns),
        "importance_note": importance_note,
    }
