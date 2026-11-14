import os
import json
import re
import hashlib
from typing import Any, Dict, List

import requests


def _num(value):
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value):
    x = _num(value)
    if x is None:
        return str(value)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.2f}"


def payload_signature(payload: Dict[str, Any]) -> str:
    """Create a stable signature of the computed results used by AI Insights.

    This prevents Streamlit session state from showing insights generated for an
    older ML run, cluster count, or multimodal summary after the underlying
    dashboard results have changed.
    """
    def normalise(obj):
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict(orient="records")
            except TypeError:
                return obj.to_dict()
        if isinstance(obj, dict):
            return {str(k): normalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
        if isinstance(obj, (list, tuple)):
            return [normalise(v) for v in obj]
        try:
            if hasattr(obj, "item"):
                return obj.item()
        except Exception:
            pass
        return obj

    raw = json.dumps(normalise(payload), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cluster_insights(mm: Dict[str, Any]) -> List[str]:
    out = []
    summary = mm.get("summary")
    if summary is None or getattr(summary, "empty", True):
        return out

    if "Sales" in summary.columns:
        row = summary.loc[summary["Sales"].idxmax()]
        out.append(
            f"Visual cluster {int(row['cluster'])} has the highest average sales "
            f"({_fmt(row['Sales'])}) across the matched products."
        )
    if "Profit" in summary.columns:
        row = summary.loc[summary["Profit"].idxmax()]
        out.append(
            f"Visual cluster {int(row['cluster'])} has the highest average profit "
            f"({_fmt(row['Profit'])})."
        )
    if "Price" in summary.columns:
        row = summary.loc[summary["Price"].idxmax()]
        out.append(
            f"Visual cluster {int(row['cluster'])} has the highest average price "
            f"({_fmt(row['Price'])})."
        )
    if "Quantity_Sold" in summary.columns:
        row = summary.loc[summary["Quantity_Sold"].idxmax()]
        out.append(
            f"Visual cluster {int(row['cluster'])} has the highest average quantity sold "
            f"({_fmt(row['Quantity_Sold'])})."
        )
    return out


def _local_insights(payload: Dict[str, Any]) -> Dict[str, Any]:
    analysis = payload.get("dataset", {})
    ml = payload.get("ml", {})
    vision = payload.get("vision", {})
    mm = payload.get("multimodal", {})

    findings = []
    recommendations = []
    warnings = []

    rows = analysis.get("rows", 0)
    cols = analysis.get("columns", 0)
    missing = analysis.get("missing_cells", 0)
    duplicates = analysis.get("duplicates", 0)

    findings.append(f"The dataset contains {rows:,} records across {cols:,} columns.")
    findings.append(
        f"The dataset contains {analysis.get('numeric_count', 0)} numeric and "
        f"{analysis.get('categorical_count', 0)} non-numeric columns."
    )

    if missing:
        warnings.append(
            f"There are {missing:,} missing cells. Review these fields before using "
            "the dataset for production-level predictions."
        )
    else:
        findings.append("No missing values were detected in the uploaded dataset.")

    if duplicates:
        warnings.append(
            f"{duplicates:,} duplicate rows were detected and should be reviewed."
        )
    else:
        findings.append("No duplicate rows were detected.")

    variance_metric = analysis.get("highest_variance_metric")
    if variance_metric:
        findings.append(
            f"{variance_metric} shows the highest variance among the analyzed numeric "
            "business metrics, indicating comparatively large differences between records."
        )

    metrics = ml.get("metrics", [])
    best_model = ml.get("best_model")
    if metrics and best_model:
        score_name = "R2" if any("R2" in r for r in metrics) else "Accuracy"
        best_score = None
        for row in metrics:
            if row.get("Model") == best_model:
                best_score = row.get(score_name)
                break
        if best_score is not None:
            if score_name == "R2":
                findings.append(
                    f"{best_model} is the strongest tested model with an R² of "
                    f"{float(best_score):.4f}, explaining a substantial share of the "
                    "variation in the selected target."
                )
            else:
                findings.append(
                    f"{best_model} is the strongest tested model with an accuracy of "
                    f"{float(best_score):.4f}."
                )

    if vision.get("images", 0):
        findings.append(
            f"{vision['images']} product images were converted into visual embeddings "
            f"and grouped into {vision.get('clusters', 0)} visual clusters."
        )

    findings.extend(_cluster_insights(mm))

    # Recommendations are based only on metrics actually present in the current summary.
    summary = mm.get("summary")
    if summary is not None and not getattr(summary, "empty", True):
        if "Sales" in summary.columns and "Profit" in summary.columns:
            best_sales = summary.loc[summary["Sales"].idxmax()]
            best_profit = summary.loc[summary["Profit"].idxmax()]
            if int(best_sales["cluster"]) != int(best_profit["cluster"]):
                recommendations.append(
                    f"Investigate visual cluster {int(best_sales['cluster'])}: it leads in "
                    "average sales, but the highest-profit cluster is different. This "
                    "suggests that sales volume and profitability may not be driven by "
                    "the same visual product characteristics."
                )
            else:
                recommendations.append(
                    f"Prioritize products in visual cluster {int(best_profit['cluster'])} "
                    "for further analysis because the same cluster leads both average "
                    "sales and average profit."
                )

        # Avoid implying that high sales alone proves an inventory problem.
        if "Stock_Available" in summary.columns and "Sales" in summary.columns:
            high_sales = summary.loc[summary["Sales"].idxmax()]
            recommendations.append(
                f"Review stock levels for visual cluster {int(high_sales['cluster'])} "
                "alongside its sales performance to determine whether inventory should "
                "be increased, maintained, or optimized."
            )

    if variance_metric:
        recommendations.append(
            f"Investigate the spread of {variance_metric}; segmenting this metric by "
            "category or visual cluster may reveal useful business differences."
        )

    if missing:
        recommendations.append(
            "Clean the missing values before final model deployment so that data-quality "
            "issues do not affect downstream analysis."
        )

    return {
        "summary": (
            "InsightVision AI combined the uploaded business data, predictive-model "
            "results, and available visual clusters to identify the strongest patterns "
            "and practical areas for investigation."
        ),
        "findings": findings,
        "recommendations": recommendations,
        "warnings": warnings,
        "source": "local_computed_results",
    }


def _try_llm(payload: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    """Use the optional LLM, but only as a wording layer over authoritative results."""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return local

    prompt = f"""
You are the business intelligence layer of InsightVision AI.
Use ONLY the supplied computed results. Do not invent numbers, change cluster IDs,
or recalculate metrics. Every numeric value and cluster ID in your response must
come directly from the supplied results. Return concise business-oriented insights.

Computed results:
{json.dumps(payload, default=str)}

Return JSON with exactly these keys:
summary: one concise executive paragraph
findings: list of 4-8 concrete findings
recommendations: list of 3-5 actionable recommendations
warnings: list of data-quality/model caveats
"""

    try:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash:generateContent"
        )
        response = requests.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.I)
        result = json.loads(text)
        result["source"] = "llm_from_computed_results"
        return result
    except Exception:
        return local


def generate_insights(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights from the exact results currently supplied by the dashboard."""
    local = _local_insights(payload)
    return _try_llm(payload, local)
