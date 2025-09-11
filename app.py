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

# ==========================
# Helpers — scales & stats
# ==========================

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

def ci_95(series_pct: pd.Series) -> Tuple[float, float, float]:
    clean = pd.to_numeric(series_pct, errors="coerce").dropna()
    n = len(clean)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = clean.mean()
    sd = clean.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else np.nan
    z = 1.96
    return (mean, mean - z * se, mean + z * se)

# ==========================
# Helpers — wide→long pairing
# ==========================
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
        prefix = parts[0].strip().lower()
        stem_raw = parts[1]
        return prefix, clean_stem(stem_raw)
    return "", clean_stem(col)

def auto_pair_columns(columns: List[str]) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    imp_map: Dict[str, str] = {}
    sat_map: Dict[str, str] = {}
    issues: List[str] = []
    for c in columns:
        prefix, stem = split_prefix_stem(str(c))
        if prefix in ("imp", "importance"):
            if stem in imp_map:
                issues.append(f"Duplicate importance stem '{stem}' for {imp_map[stem]} and {c}")
            imp_map[stem] = c
        elif prefix in ("sat", "satisfaction"):
            if stem in sat_map:
                issues.append(f"Duplicate satisfaction stem '{stem}' for {sat_map[stem]} and {c}")
            sat_map[stem] = c
    return imp_map, sat_map, issues

def reshape_wide_manual(df: pd.DataFrame, respondent_id_col: str, imp_cols: List[str], sat_cols: List[str]) -> pd.DataFrame:
    if len(imp_cols) != len(sat_cols):
        raise ValueError("Importance and Satisfaction column blocks must be the same length.")
    frames = []
    for imp_col, sat_col in zip(imp_cols, sat_cols):
        stem = clean_stem(imp_col)
        part = pd.DataFrame({
            "respondent_id": df[respondent_id_col],
            "outcome_id": stem,
            "importance": df[imp_col],
            "satisfaction": df[sat_col],
        })
        frames.append(part)
    return pd.concat(frames, ignore_index=True)

def reshape_wide_autopair(df: pd.DataFrame, respondent_id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    imp_map, sat_map, issues = auto_pair_columns(df.columns.tolist())
    stems = sorted(set(imp_map.keys()) | set(sat_map.keys()))
    pairs = []
    pairing_rows = []
    for stem in stems:
        imp_c = imp_map.get(stem, None)
        sat_c = sat_map.get(stem, None)
        pairing_rows.append({
            "stem": stem,
            "importance_col": imp_c,
            "satisfaction_col": sat_c,
            "ok": imp_c is not None and sat_c is not None
        })
        if imp_c is None:
            issues.append(f"Missing importance column for stem '{stem}'")
        if sat_c is None:
            issues.append(f"Missing satisfaction column for stem '{stem}'")
        if imp_c is not None and sat_c is not None:
            pairs.append((stem, imp_c, sat_c))
    pairing_table = pd.DataFrame(pairing_rows)
    frames = []
    for stem, imp_col, sat_col in pairs:
        frames.append(pd.DataFrame({
            "respondent_id": df[respondent_id_col],
            "outcome_id": stem,
            "importance": df[imp_col],
            "satisfaction": df[sat_col],
        }))
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["respondent_id","outcome_id","importance","satisfaction"])
    return long_df, pairing_table, issues

# ==========================
# Aggregation
# ==========================
def aggregate(df_long: pd.DataFrame, agg_mode: str, formula: str) -> pd.DataFrame:
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

    def opp_calc(imp_mean, sat_mean, formula):
        if formula == "Ulwick classic":
            return imp_mean + (imp_mean - sat_mean)
        elif formula == "Weighted gap":
            return 2 * imp_mean - sat_mean
        elif formula == "Capped gap":
            return imp_mean + max(imp_mean - sat_mean, 0)
        return np.nan

    rows = []
    for outcome, g in data.groupby("outcome_id"):
        n = len(g)
        imp_mean, imp_lo, imp_hi = ci_95(g["imp_pct"])
        sat_mean, sat_lo, sat_hi = ci_95(g["sat_pct"])
        opp = opp_calc(imp_mean, sat_mean, formula)
        rows.append({
            "outcome_id": outcome,
            "N": n,
            "Importance (0-100)": round(imp_mean, 2),
            "Importance 95% CI": f"[{round(imp_lo,2)}, {round(imp_hi,2)}]",
            "Satisfaction (0-100)": round(sat_mean, 2),
            "Satisfaction 95% CI": f"[{round(sat_lo,2)}, {round(sat_hi,2)}]",
            "Opportunity": round(opp, 2),
        })

    out = pd.DataFrame(rows).sort_values("Opportunity", ascending=False, na_position="last").reset_index(drop=True)
    if dropped > 0:
        st.info(f"Dropped {dropped} rows outside the 1–10 range or non-numeric.")
    return out

# ==========================
# Charts & downloads
# ==========================
def make_bubble(df_agg: pd.DataFrame, color_hex: str):
    if df_agg.empty:
        return None
    chart = (
        alt.Chart(df_agg)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("Satisfaction (0-100)", title="Satisfaction (0–100)"),
            y=alt.Y("Importance (0-100)", title="Importance (0–100)"),
            size=alt.Size("Opportunity", scale=alt.Scale(type="sqrt"), legend=alt.Legend(title="Opportunity")),
            color=alt.value(color_hex),
            tooltip=["outcome_id", "N", "Importance (0-100)", "Satisfaction (0-100)", "Opportunity"],
        )
        .properties(height=500)
    )
    rules = alt.Chart(pd.DataFrame({"x":[50]})).mark_rule(strokeDash=[4,4]).encode(x="x") | \
            alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(strokeDash=[4,4]).encode(y="y")
    return chart + rules

def download_chart(chart: alt.Chart, kind: str) -> bytes:
    spec = chart.to_dict()
    if kind == "PNG":
        return vegalite_to_png(spec, scale=2)
    return vegalite_to_svg(spec)

# ==========================
# UI
# ==========================
st.title("Opportunity Score Calculator")

with st.expander("About this app"):
    st.markdown(
        """
**Scales and formulas**  
- Inputs must be **1–10**. Choose aggregation: **Means** or **Top-2-Box (≥9)**.  
- We convert both to **0–100** so formulas are comparable.  
- Formulas:  
  • **Ulwick classic**: Opp = Imp + (Imp − Sat)  
  • **Weighted gap**: Opp = 2×Imp − Sat  
  • **Capped gap**: Opp = Imp + max(Imp − Sat, 0)  

**Data schema (long)**  
`respondent_id, outcome_id, importance, satisfaction`  
Values must be numeric 1–10.
        """
    )

    # Template downloads
    long_template = pd.DataFrame({
        "respondent_id": ["R001", "R001", "R002"],
        "outcome_id": ["login", "search", "login"],
        "importance": [9, 8, 7],
        "satisfaction": [6, 7, 8],
    })
    buf_long = io.StringIO(); long_template.to_csv(buf_long, index=False)
    st.download_button("Download long-format CSV template", buf_long.getvalue().encode("utf-8"),
                       file_name="oppscore_long_template.csv", mime="text/csv")

    wide_template = pd.DataFrame({
        "respondent_id": ["R001", "R002"],
        "imp_login": [9,7], "imp_search": [8,9], "imp_reporting": [10,6],
        "sat_login": [6,8], "sat_search": [7,6], "sat_reporting": [5,5],
    })
    buf_wide = io.StringIO(); wide_template.to_csv(buf_wide, index=False)
    st.download_button("Download wide-format CSV template", buf_wide.getvalue().encode("utf-8"),
                       file_name="oppscore_wide_template.csv", mime="text/csv")

st.sidebar.header("1) Upload data")
upload = st.sidebar.file_uploader("CSV only", type=["csv"])

mode = st.sidebar.radio("Data shape", ["Long format", "Wide format"], index=0)
agg_mode = st.sidebar.radio("Aggregation", ["Means (1–10)", "Top-2-Box (≥9)"])
formula = st.sidebar.selectbox("Opportunity formula", ["Ulwick classic", "Weighted gap", "Capped gap"], index=0)
color_hex = st.sidebar.text_input("Bubble color (hex)", value="#3F51B5")

if upload is not None:
    df = pd.read_csv(upload)
    st.write("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    if mode == "Long format":
        cols = df.columns.tolist()
        rid = st.selectbox("respondent_id column", cols, index=0)
        oid = st.selectbox("outcome_id column", cols, index=1)
        imp = st.selectbox("importance column (1–10)", cols, index=2)
        sat = st.selectbox("satisfaction column (1–10)", cols, index=3)
        long_df = df[[rid, oid, imp, sat]].rename(columns={
            rid:"respondent_id", oid:"outcome_id", imp:"importance", sat:"satisfaction"
        })

    else:  # Wide format
        cols = df.columns.tolist()
        rid = st.selectbox("respondent_id column", cols, index=0)

        st.markdown("#### Pairing options")
        use_auto = st.checkbox("Auto-match by prefixes (imp_/sat_, importance_/satisfaction_)", value=True)

        long_df = None
        if use_auto:
            long_df, pairing_table, issues = reshape_wide_autopair(df, rid)
            st.markdown("**Proposed pairing**")
            st.dataframe(pairing_table, use_container_width=True)
            if issues:
                st.warning("\\n".join(issues))
            if long_df is not None and not long_df.empty:
                st.markdown("Preview (first 30 long rows):")
                st.dataframe(long_df.head(30), use_container_width=True)
        else:
            st.markdown("Select the Importance and Satisfaction blocks (same order & length):")
            imp_cols = st.multiselect("Importance columns (1–10)", cols)
            sat_cols = st.multiselect("Satisfaction columns (1–10)", cols)
            if imp_cols and sat_cols:
                try:
                    long_df = reshape_wide_manual(df, rid, imp_cols, sat_cols)
                    st.markdown("Preview (first 30 long rows):")
                    st.dataframe(long_df.head(30), use_container_width=True)
                except Exception as e:
                    st.error(str(e))

    if 'long_df' in locals() and long_df is not None and not long_df.empty:
        st.markdown("### Aggregated results")
        agg = aggregate(long_df, agg_mode, formula)
        st.dataframe(agg, use_container_width=True)

        csv_text = agg.to_csv(index=False)
        st.download_button("Download aggregated CSV", data=csv_text.encode("utf-8"),
                           file_name="opportunity_aggregated.csv", mime="text/csv")

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
else:
    st.info("Upload a CSV to begin. Use the templates in About if needed.")
