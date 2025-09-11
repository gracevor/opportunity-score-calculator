import io
import json
import math
import re
from typing import List, Tuple, Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from vl_convert import vegalite_to_png, vegalite_to_svg

st.set_page_config(page_title="Opportunity Score Calculator", layout="wide")

# =====================================
# Helpers: scales, validation, stats
# =====================================

def to_0_100_from_mean_1_to_10(x: pd.Series) -> pd.Series:
    return (pd.to_numeric(x, errors="coerce") - 1.0) / 9.0 * 100.0

def to_0_100_from_t2b_bool(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype(float) * 100.0

def compute_t2b_flags_1_to_10(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.ge(9.0)

def valid_1_to_10(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.between(1, 10)

def ci_95(series_pct: pd.Series):
    clean = pd.to_numeric(series_pct, errors="coerce").dropna()
    n = len(clean)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = clean.mean()
    sd = clean.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    z = 1.96
    return (mean, mean - z * se, mean + z * se)

# =====================================
# Helpers: wide to long auto-detection
# =====================================
PREFIX_SET = {"imp", "importance", "sat", "satisfaction"}
DELIMS = r"[:_\- ]+"

def clean_stem(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]

def split_prefix_stem(col: str) -> Tuple[str, str]:
    parts = re.split(DELIMS, col, maxsplit=1)
    if len(parts) == 2 and parts[0].strip().lower() in PREFIX_SET:
        return parts[0].strip().lower(), clean_stem(parts[1])
    return "", clean_stem(col)

def auto_pair_columns(columns: List[str]):
    imp_map: Dict[str, str] = {}
    sat_map: Dict[str, str] = {}
    for c in columns:
        prefix, stem = split_prefix_stem(str(c))
        if prefix in ("imp", "importance"):
            imp_map[stem] = c
        elif prefix in ("sat", "satisfaction"):
            sat_map[stem] = c
    stems = sorted(set(imp_map.keys()) & set(sat_map.keys()))
    return stems, imp_map, sat_map

def reshape_wide_autopair(df: pd.DataFrame, respondent_id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stems, imp_map, sat_map = auto_pair_columns(df.columns.tolist())
    frames = []
    for stem in stems:
        frames.append(pd.DataFrame({
            "respondent_id": df[respondent_id_col],
            "outcome_id": stem,
            "importance": df[imp_map[stem]],
            "satisfaction": df[sat_map[stem]],
        }))
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["respondent_id", "outcome_id", "importance", "satisfaction"]
    )
    pairing = pd.DataFrame({
        "outcome_id": stems,
        "importance_col": [imp_map[s] for s in stems],
        "satisfaction_col": [sat_map[s] for s in stems],
    })
    return long_df, pairing

# =====================================
# Aggregation and opportunity formulas
# =====================================

def aggregate(df_long: pd.DataFrame, agg_mode: str, formula: str, label_map: Dict[str, str]) -> pd.DataFrame:
    mask_imp = valid_1_to_10(df_long["importance"])
    mask_sat = valid_1_to_10(df_long["satisfaction"])
    valid = mask_imp & mask_sat
    dropped = int((~valid).sum())
    data = df_long.loc[valid].copy()

    if agg_mode == "Means (1–10)":
        data["imp_pct"] = to_0_100_from_mean_1_to_10(data["importance"])
        data["sat_pct"] = to_0_100_from_mean_1_to_10(data["satisfaction"])
    else:  # Top-2-Box (≥9)
        data["imp_pct"] = to_0_100_from_t2b_bool(compute_t2b_flags_1_to_10(data["importance"]))
        data["sat_pct"] = to_0_100_from_t2b_bool(compute_t2b_flags_1_to_10(data["satisfaction"]))

    def opp_calc(imp_mean, sat_mean):
        if formula == "Ulwick classic":
            return imp_mean + (imp_mean - sat_mean)
        if formula == "Weighted gap":
            return 2 * imp_mean - sat_mean
        if formula == "Capped gap":
            return imp_mean + max(imp_mean - sat_mean, 0)
        return np.nan

    rows = []
    for outcome, g in data.groupby("outcome_id"):
        n = len(g)
        imp_mean, imp_lo, imp_hi = ci_95(g["imp_pct"])
        sat_mean, sat_lo, sat_hi = ci_95(g["sat_pct"])
        opp = opp_calc(imp_mean, sat_mean)
        rows.append({
            "outcome_id": outcome,
            "label": label_map.get(str(outcome), str(outcome)),
            "N": n,
            "Importance (0-100)": round(imp_mean, 2),
            "Importance 95% CI": f"[{round(imp_lo,2)}, {round(imp_hi,2)}]",
            "Satisfaction (0-100)": round(sat_mean, 2),
            "Satisfaction 95% CI": f"[{round(sat_lo,2)}, {round(sat_hi,2)}]",
            "Opportunity": round(opp, 2),
        })

    out = pd.DataFrame(rows).sort_values(
        ["Opportunity", "label"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    if dropped > 0:
        st.info(f"Dropped {dropped} rows outside the 1–10 range or non-numeric.")
    return out

# =====================================
# Charts and downloads
# =====================================

def make_bubble(df_agg: pd.DataFrame, color_hex: str):
    if df_agg.empty:
        return None
    chart = (
        alt.Chart(df_agg)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Satisfaction (0-100)", title="Satisfaction (0–100)"),
            y=alt.Y("Importance (0-100)", title="Importance (0–100)"),
            size=alt.Size("Opportunity", scale=alt.Scale(type="sqrt"), legend=alt.Legend(title="Opportunity")),
            color=alt.value(color_hex),
            tooltip=["label", "outcome_id", "N", "Importance (0-100)", "Satisfaction (0-100)", "Opportunity"],
        )
        .properties(height=520)
    )
    rules = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 4]).encode(x="x") | \
            alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[4, 4]).encode(y="y")
    return chart + rules

def download_chart(chart: alt.Chart, kind: str) -> bytes:
    spec = chart.to_dict()
    if kind == "PNG":
        return vegalite_to_png(spec, scale=2)
    return vegalite_to_svg(spec)

# =====================================
# UI: simple, no column pickers
# =====================================

st.title("Opportunity Score Calculator")

with st.expander("How to format your files"):
    st.markdown(
        """
**This app auto-detects format. Follow these header rules.**

**Option A — Long format (recommended)**  
Columns must be exactly:  
`respondent_id, outcome_id, importance, satisfaction`  
All values are 1–10.

**Option B — Wide format**  
- One column: `respondent_id`  
- Importance columns start with `imp_` or `importance_`, for example `imp_outcome_1`  
- Satisfaction columns start with `sat_` or `satisfaction_`, for example `sat_outcome_1`  
- The stem after the prefix (for example `outcome_1`) must be identical for the pair  
- Stems become your `outcome_id`

**Optional label mapping (separate CSV)**  
Upload a second CSV with columns:  
`outcome_id,label`  
Example:  
`outcome_1, Reduce mean time to fix vulnerabilities`  
If provided, labels appear in the table, tooltips, and a legend below the chart.
        """
    )

    # Templates
    long_template = pd.DataFrame({
        "respondent_id": ["R001", "R001", "R002"],
        "outcome_id": ["outcome_1", "outcome_2", "outcome_1"],
        "importance": [9, 8, 7],
        "satisfaction": [6, 7, 8],
    })
    buf_long = io.StringIO(); long_template.to_csv(buf_long, index=False)
    st.download_button(
        "Download long-format CSV template",
        buf_long.getvalue().encode("utf-8"),
        file_name="oppscore_long_template.csv",
        mime="text/csv",
    )

    wide_template = pd.DataFrame({
        "respondent_id": ["R001", "R002"],
        "imp_outcome_1": [9, 7], "imp_outcome_2": [8, 9],
        "sat_outcome_1": [6, 8], "sat_outcome_2": [7, 6],
    })
    buf_wide = io.StringIO(); wide_template.to_csv(buf_wide, index=False)
    st.download_button(
        "Download wide-format CSV template",
        buf_wide.getvalue().encode("utf-8"),
        file_name="oppscore_wide_template.csv",
        mime="text/csv",
    )

    map_template = pd.DataFrame({
        "outcome_id": ["outcome_1", "outcome_2"],
        "label": ["Security: reduce MTTR", "Search: improve relevance"],
    })
    buf_map = io.StringIO(); map_template.to_csv(buf_map, index=False)
    st.download_button(
        "Download outcome label mapping template",
        buf_map.getvalue().encode("utf-8"),
        file_name="oppscore_label_map.csv",
        mime="text/csv",
    )

st.sidebar.header("Upload your data")
data_file = st.sidebar.file_uploader("Survey data CSV (long or wide)", type=["csv"])
map_file = st.sidebar.file_uploader("Optional: outcome label mapping CSV", type=["csv"])

agg_mode = st.sidebar.radio("Aggregation", ["Means (1–10)", "Top-2-Box (≥9)"])
formula = st.sidebar.selectbox("Opportunity formula", ["Ulwick classic", "Weighted gap", "Capped gap"], index=0)
color_hex = st.sidebar.text_input("Bubble color (hex)", value="#3F51B5")

label_map: Dict[str, str] = {}
if map_file is not None:
    try:
        df_map = pd.read_csv(map_file, dtype=str)
        cols_norm = {c.lower(): c for c in df_map.columns}
        if {"outcome_id", "label"}.issubset(set(cols_norm.keys())):
            df_map = df_map.rename(columns={v: k for k, v in cols_norm.items()})
            label_map = dict(zip(df_map["outcome_id"].astype(str), df_map["label"].astype(str)))
        else:
            st.sidebar.warning("Mapping CSV must have columns: outcome_id,label")
    except Exception as e:
        st.sidebar.error(f"Failed to read mapping CSV: {e}")

if data_file is not None:
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.write("Preview (first 10 rows):")
    st.dataframe(df.head(10), use_container_width=True)

    cols_lower = {c.lower() for c in df.columns}
    long_required = {"respondent_id", "outcome_id", "importance", "satisfaction"}

    if long_required.issubset(cols_lower):
        # Long format
        colmap = {name: [c for c in df.columns if c.lower() == name][0] for name in long_required}
        long_df = df[[colmap["respondent_id"], colmap["outcome_id"], colmap["importance"], colmap["satisfaction"]]]
        long_df = long_df.rename(columns={
            colmap["respondent_id"]: "respondent_id",
            colmap["outcome_id"]: "outcome_id",
            colmap["importance"]: "importance",
            colmap["satisfaction"]: "satisfaction",
        })
        st.success("Detected long format.")
    else:
        # Wide format
        rid_candidates = [c for c in df.columns if c.lower() == "respondent_id"]
        if not rid_candidates:
            st.error("Wide format requires a 'respondent_id' column.")
            st.stop()
        rid = rid_candidates[0]
        long_df, pairing = reshape_wide_autopair(df, rid)
        if long_df.empty:
            st.error("Could not auto-detect any imp_/sat_ pairs. Check your headers.")
            st.dataframe(pd.DataFrame({"hint": ["Use imp_outcome_1 and sat_outcome_1 style headers"]}))
            st.stop()
        st.success("Detected wide format and auto-paired columns.")
        st.markdown("**Detected pairs**")
        st.dataframe(pairing, use_container_width=True)

    st.markdown("### Aggregated results")
    agg = aggregate(long_df, agg_mode, formula, label_map)
    if "label" in agg.columns:
        order = [
            "label", "outcome_id", "N",
            "Importance (0-100)", "Importance 95% CI",
            "Satisfaction (0-100)", "Satisfaction 95% CI",
            "Opportunity",
        ]
        agg = agg[[c for c in order if c in agg.columns]]
    st.dataframe(agg, use_container_width=True)

    csv_text = agg.to_csv(index=False)
    st.download_button(
        "Download aggregated CSV",
        data=csv_text.encode("utf-8"),
        file_name="opportunity_aggregated.csv",
        mime="text/csv",
    )

    st.markdown("### Bubble chart")
    chart = make_bubble(agg, color_hex)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            png_bytes = download_chart(chart, "PNG")
            st.download_button("Download PNG", data=png_bytes, file_name="bubble.png", mime="image/png")
        with col2:
            svg_bytes = download_chart(chart, "SVG")
            st.download_button("Download SVG", data=svg_bytes, file_name="bubble.svg", mime="image/svg+xml")

    if label_map:
        st.markdown("#### Outcome legend")
        legend_df = pd.DataFrame(sorted(label_map.items()), columns=["outcome_id", "label"])
        st.dataframe(legend_df, use_container_width=True)
else:
    st.info("Upload a CSV to begin. Use the templates above if needed.")
