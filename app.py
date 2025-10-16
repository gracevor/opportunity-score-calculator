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
    return s.between(1, 5, inclusive='both')

def calculate_top2_percentage_times_10(series: pd.Series) -> float:
    """Calculate (count of 4s and 5s / total count) * 10"""
    s = pd.to_numeric(series, errors="coerce")
    # Only count valid responses in 1-5 range (excludes NaN/null)
    valid_responses = s.between(1, 5, inclusive='both')
    if valid_responses.sum() == 0:
        return 0.0
    top2_count = s.ge(4.0).sum()  # This already excludes NaN values
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
    # Process each outcome separately to handle missing data properly
    rows = []
    calculation_details = []
    total_rows_processed = 0
    total_rows_used = 0
    
    for outcome, g in df_long.groupby("outcome_id"):
        # For this outcome, only use rows where BOTH importance and satisfaction are valid (1-5)
        imp_numeric = pd.to_numeric(g["importance"], errors="coerce")
        sat_numeric = pd.to_numeric(g["satisfaction"], errors="coerce")
        
        imp_valid = imp_numeric.between(1, 5, inclusive='both')
        sat_valid = sat_numeric.between(1, 5, inclusive='both')
        both_valid = imp_valid & sat_valid
        
        # Count statistics for this outcome
        total_responses_outcome = len(g)
        valid_responses_outcome = both_valid.sum()
        dropped_responses_outcome = total_responses_outcome - valid_responses_outcome
        
        total_rows_processed += total_responses_outcome
        total_rows_used += valid_responses_outcome
        
        if valid_responses_outcome == 0:
            # No valid data for this outcome
            rows.append({
                "label": label_map.get(str(outcome), str(outcome)),
                "outcome_id": outcome,
                "N": 0,
                "Importance": 0.0,
                "Satisfaction": 0.0,
                "Opportunity": 0.0,
            })
            calculation_details.append({
                "outcome_id": outcome,
                "total_responses": total_responses_outcome,
                "valid_pairs": valid_responses_outcome,
                "dropped": dropped_responses_outcome,
                "imp_4_5_count": 0,
                "imp_score": 0.0,
                "sat_4_5_count": 0,
                "sat_score": 0.0,
                "opportunity": 0.0,
                "formula": "No valid data pairs"
            })
            continue
        
        # Use only valid pairs for calculation
        valid_data = g[both_valid]
        
        # Calculate scores using only valid responses
        imp_score = calculate_top2_percentage_times_10(valid_data["importance"])
        sat_score = calculate_top2_percentage_times_10(valid_data["satisfaction"])
        opportunity = imp_score + (imp_score - sat_score)
        
        # Count 4s and 5s for debugging
        imp_4_5_count = (pd.to_numeric(valid_data["importance"], errors="coerce") >= 4).sum()
        sat_4_5_count = (pd.to_numeric(valid_data["satisfaction"], errors="coerce") >= 4).sum()
        
        calculation_details.append({
            "outcome_id": outcome,
            "total_responses": total_responses_outcome,
            "valid_pairs": valid_responses_outcome,
            "dropped": dropped_responses_outcome,
            "imp_4_5_count": imp_4_5_count,
            "imp_score": round(imp_score, 4),
            "sat_4_5_count": sat_4_5_count,
            "sat_score": round(sat_score, 4),
            "opportunity": round(opportunity, 4),
            "formula": f"{imp_score:.2f} + ({imp_score:.2f} - {sat_score:.2f}) = {opportunity:.2f}"
        })
        
        rows.append({
            "label": label_map.get(str(outcome), str(outcome)),
            "outcome_id": outcome,
            "N": valid_responses_outcome,
            "Importance": round(imp_score, 2),
            "Satisfaction": round(sat_score, 2),
            "Opportunity": round(opportunity, 2),
        })

    out = pd.DataFrame(rows).sort_values(
        ["Opportunity", "label"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)

    # Show overall data quality summary
    total_dropped = total_rows_processed - total_rows_used
    if total_dropped > 0:
        st.info(f"📊 **Missing Data Summary:** Used {total_rows_used:,} valid response pairs out of {total_rows_processed:,} total responses. Excluded {total_dropped:,} responses with missing or invalid data.")
    
    # Show calculation details for debugging
    if calculation_details:
        st.markdown("**Calculation Details (per outcome):**")
        calc_df = pd.DataFrame(calculation_details)
        st.dataframe(calc_df, use_container_width=True)
        
        # Show any outcomes with significant missing data
        high_missing = calc_df[calc_df["dropped"] / calc_df["total_responses"] > 0.2]  # >20% missing
        if not high_missing.empty:
            st.warning(f"⚠️ **High Missing Data Alert:** {len(high_missing)} outcomes have >20% missing responses. Consider reviewing data quality for: {', '.join(high_missing['outcome_id'].tolist())}")
    
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
    
    # Create a column for the legend that uses mapped labels if available
    df_display = df_agg.copy()
    df_display['legend_label'] = df_display.apply(
        lambda row: row['label'] if row['label'] != row['outcome_id'] else row['outcome_id'], 
        axis=1
    )
    
    # Get dynamic ranges for both bubble sizes AND axis domains
    min_opp = df_agg['Opportunity'].min()
    max_opp = df_agg['Opportunity'].max()
    
    # Get actual data ranges for axes with some padding
    min_sat = max(0, df_agg['Satisfaction'].min() - 1)  # Don't go below 0
    max_sat = min(10, df_agg['Satisfaction'].max() + 1)  # Don't go above 10
    min_imp = max(0, df_agg['Importance'].min() - 1)
    max_imp = min(10, df_agg['Importance'].max() + 1)
    
    # Create the chart with dynamic axes that fit your data
    chart = (
        alt.Chart(df_display)
        .mark_circle(opacity=0.75, stroke='white', strokeWidth=1)
        .encode(
            x=alt.X(
                "Satisfaction", 
                title="Satisfaction (0–10)", 
                scale=alt.Scale(domain=[min_sat, max_sat], nice=True)  # Dynamic domain with padding
            ),
            y=alt.Y(
                "Importance", 
                title="Importance (0–10)", 
                scale=alt.Scale(domain=[min_imp, max_imp], nice=True)  # Dynamic domain with padding
            ),
            size=alt.Size(
                "Opportunity", 
                scale=alt.Scale(
                    type="pow", exponent=2,
                    range=[200, 3000],  # Even larger bubbles for more impact
                    domain=[min_opp, max_opp]
                ), 
                legend=None
            ),
            color=alt.Color(
                "legend_label", 
                scale=alt.Scale(range=github_colors), 
                legend=alt.Legend(
                    title="Outcomes",
                    titleFontSize=12,
                    labelFontSize=10,
                    labelLimit=200
                )
            ),
            tooltip=[
                alt.Tooltip("legend_label:N", title="Outcome"),
                alt.Tooltip("outcome_id:N", title="ID"), 
                alt.Tooltip("N:O", title="Sample Size"),
                alt.Tooltip("Importance:Q", title="Importance", format=".2f"),
                alt.Tooltip("Satisfaction:Q", title="Satisfaction", format=".2f"),
                alt.Tooltip("Opportunity:Q", title="Opportunity", format=".2f")
            ],
        )
        .properties(height=520, width=700)
    )
    
    return chart

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

**Missing Data Handling:**
- For each outcome, only respondents with BOTH importance AND satisfaction values (1-5) are included
- Missing, blank, or invalid values are excluded per outcome
- Sample sizes (N) may vary between outcomes based on response completeness
- This maximizes data usage while ensuring valid comparisons

**Optional label mapping (separate CSV)**  
Upload a second CSV with columns:  
`outcome_id,label`  
Example:  
`outcome_1, Reduce mean time to fix vulnerabilities`  
If provided, labels appear in the table, tooltips, and a legend below the chart.

**Calculation method:**
- Importance = (Count of 4s and 5s / Total valid responses) × 10
- Satisfaction = (Count of 4s and 5s / Total valid responses) × 10  
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
    
    # Add data validation summary
    st.markdown("**Data Validation Summary**")
    total_rows = len(long_df)
    
    # Check importance values
    imp_numeric = pd.to_numeric(long_df["importance"], errors="coerce")
    imp_valid_count = imp_numeric.between(1, 5).sum()
    imp_invalid_count = total_rows - imp_valid_count
    
    # Check satisfaction values  
    sat_numeric = pd.to_numeric(long_df["satisfaction"], errors="coerce")
    sat_valid_count = sat_numeric.between(1, 5).sum()
    sat_invalid_count = total_rows - sat_valid_count
    
    # Show unique values found
    imp_unique = sorted([x for x in imp_numeric.dropna().unique() if not pd.isna(x)])
    sat_unique = sorted([x for x in sat_numeric.dropna().unique() if not pd.isna(x)])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Importance Values:**")
        st.write(f"• Valid (1-5): {imp_valid_count:,} rows")
        st.write(f"• Invalid: {imp_invalid_count:,} rows")
        st.write(f"• Unique values found: {imp_unique}")
    
    with col2:
        st.write("**Satisfaction Values:**")
        st.write(f"• Valid (1-5): {sat_valid_count:,} rows")
        st.write(f"• Invalid: {sat_invalid_count:,} rows")  
        st.write(f"• Unique values found: {sat_unique}")
    
    if imp_invalid_count > 0 or sat_invalid_count > 0:
        st.warning(f"⚠️ Found invalid values. Check your data - all importance and satisfaction values must be between 1 and 5 (integers or decimals).")

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
