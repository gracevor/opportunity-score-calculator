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

def valid_1_to_5(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.between(1, 5)

def calculate_top2_percentage_times_10(series: pd.Series) -> float:
    """Calculate (count of 4s and 5s / total count) * 10"""
    s = pd.to_numeric(series, errors="coerce")
    valid_responses = s.between(1, 5)
    if valid_responses.sum() == 0:
        return 0.0
    top2_count = s.ge(4.0).sum()
    total_count = valid_responses.sum()
    return (top2_count / total_count) * 10

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
# Aggregation and opportunity calculation
# =====================================

def aggregate(df_long: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    mask_imp = valid_1_to_5(df_long["importance"])
    mask_sat = valid_1_to_5(df_long["satisfaction"])
    valid = mask_imp & mask_sat
    dropped = int((~valid).sum())
    data = df_long.loc[valid].copy()

    rows = []
    for outcome, g in data.groupby("outcome_id"):
        n = len(g)
        imp_score = calculate_top2_percentage_times_10(g["importance"])
        sat_score = calculate_top2_percentage_times_10(g["satisfaction"])
        opportunity = imp_score + (imp_score - sat_score)
        
        rows.append({
            "label": label_map.get(str(outcome), str(outcome)),
            "outcome_id": outcome,
            "N": n,
            "Importance": round(imp_score, 2),
            "Satisfaction": round(sat_score, 2),
            "Opportunity": round(opportunity, 2),
        })

    out = pd.DataFrame(rows).sort_values(
        ["Opportunity", "label"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    if dropped > 0:
        st.info(f"Dropped {dropped} rows outside the 1–5 range or with non-numeric values.")
    return out

# =====================================
# Charts and downloads
# =====================================

def make_bubble(df_agg: pd.DataFrame):
    if df_agg.empty:
        return None
    
    # GitHub brand colors
    github_colors = [
        "#096BDE", "#000AFF", "#6BD6D0", "#A9E500", "#FF507A", 
        "#FFA6D6", "#8250DF", "#FF5934", "#2DA44E", "#00FF46", 
        "#7C72FF", "#F4E162", "#5F00FF"
    ]
    
    chart = (
        alt.Chart(df_agg)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("Satisfaction", title="Satisfaction (0–10)", scale=alt.Scale(domain=[0, 10])),
            y=alt.Y("Importance", title="Importance (0–10)", scale=alt.Scale(domain=[0, 10])),
            size=alt.Size("Opportunity", scale=alt.Scale(type="sqrt"), legend=alt.Legend(title="Opportunity")),
            color=alt.Color("outcome_id", scale=alt.Scale(range=github_colors), legend=alt.Legend(title="Outcome")),
            tooltip=["label", "outcome_id", "N", "Importance", "Satisfaction", "Opportunity"],
        )
        .properties(height=520)
    )
    # Reference lines at 1.0 as requested
    rules = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 4]).encode(x="x") | \
            alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4]).encode(y="y")
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
**This app requires wide format CSV files.**

**Wide format requirements:**  
- One column: `respondent_id`  
- Importance columns start with `imp_` or `importance_`, for example `imp_outcome_1`  
- Satisfaction columns start with `sat_` or `satisfaction_`, for example `sat_outcome_1`  
- The stem after the prefix (for example `outcome_1`) must be identical for the pair  
- Stems become your `outcome_id`
- **All importance and satisfaction values must be on a 1–5 scale**

**Optional label mapping (separate CSV)**  
Upload a second CSV with columns:  
`outcome_id,label`  
Example:  
`outcome_1, Reduce mean time to fix vulnerabilities`  
If provided, labels appear in the table, tooltips, and a legend below the chart.

**Calculation method:**
- Importance = (Count of 4s and 5s / Total responses) × 10
- Satisfaction = (Count of 4s and 5s / Total responses) × 10  
- Opportunity = Importance + (Importance - Satisfaction)
        """
    )

    # Templates
    wide_template = pd.DataFrame({
        "respondent_id": ["R001", "R002", "R003"],
        "imp_outcome_1": [5, 4, 3], "imp_outcome_2": [4, 5, 4],
        "sat_outcome_1": [3, 4, 5], "sat_outcome_2": [2, 3, 4],
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
data_file = st.sidebar.file_uploader("Survey data CSV (wide format)", type=["csv"])
map_file = st.sidebar.file_uploader("Optional: outcome label mapping CSV", type=["csv"])

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

    # Only support wide format
    rid_candidates = [c for c in df.columns if c.lower() == "respondent_id"]
    if not rid_candidates:
        st.error("Wide format requires a 'respondent_id' column.")
        st.stop()
    
    rid = rid_candidates[0]
    long_df, pairing = reshape_wide_autopair(df, rid)
    if long_df.empty:
        st.error("Could not auto-detect any imp_/sat_ pairs. Check your headers.")
        st.dataframe(pd.DataFrame({"hint": ["Use imp_outcome_1 and sat_outcome_1 style headers with 1-5 scale values"]}))
        st.stop()
    
    st.success("Detected wide format and auto-paired columns.")
    st.markdown("**Detected pairs**")
    st.dataframe(pairing, use_container_width=True)

    st.markdown("### Results")
    agg = aggregate(long_df, label_map)
    st.dataframe(agg, use_container_width=True)

    csv_text = agg.to_csv(index=False)
    st.download_button(
        "Download results as CSV",
        data=csv_text.encode("utf-8"),
        file_name="opportunity_results.csv",
        mime="text/csv",
    )

    st.markdown("### Opportunity Score Bubble Chart")
    chart = make_bubble(agg)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            png_bytes = download_chart(chart, "PNG")
            st.download_button("Download chart as PNG", data=png_bytes, file_name="opportunity_chart.png", mime="image/png")
        with col2:
            svg_bytes = download_chart(chart, "SVG")
            st.download_button("Download chart as SVG", data=svg_bytes, file_name="opportunity_chart.svg", mime="image/svg+xml")

    if label_map:
        st.markdown("#### Outcome legend")
        legend_df = pd.DataFrame(sorted(label_map.items()), columns=["outcome_id", "label"])
        st.dataframe(legend_df, use_container_width=True)
else:
    st.info("Upload a wide-format CSV to begin. Use the template above if needed.")
