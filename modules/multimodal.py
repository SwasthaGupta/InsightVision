from pathlib import Path
import pandas as pd
import re


def _tokens(name):
    return set(x.lower() for x in re.findall(r"[A-Za-z0-9]+", Path(name).stem))


def match_images_to_data(df, metadata):
    candidates = [c for c in df.columns if df[c].dropna().astype(str).nunique() > 1]
    best = None
    for col in candidates:
        mapping = {}
        for idx, val in df[col].items():
            mapping.setdefault(str(val).strip().lower(), idx)
        rows = []
        for image_index, item in enumerate(metadata):
            matches = [mapping[t] for t in _tokens(item["path"]) if t in mapping]
            if matches: rows.append((image_index, matches[0]))
        if best is None or len(rows) > len(best[1]): best = (col, rows)
    if not best or not best[1]: return pd.DataFrame()
    col, rows = best; records = []
    for image_index, row_index in rows:
        r = df.loc[row_index].to_dict()
        r.update({"image_index": image_index, "image_file": Path(metadata[image_index]["path"]).name,
                  "matched_id_column": col})
        records.append(r)
    return pd.DataFrame(records)


def summarize_clusters(matches, numeric_cols):
    # Identifier columns should not be averaged.
    cols = [c for c in numeric_cols if c in matches.columns and not str(c).lower().endswith("id")]
    summary = matches.groupby("cluster")[cols].mean().reset_index().round(3)
    summary.insert(1, "Products", matches.groupby("cluster").size().values)
    return summary
