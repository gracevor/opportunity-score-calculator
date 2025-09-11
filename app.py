import io
if issues:
st.warning("
".join(issues))
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


if st.sidebar.button("Save mapping to localStorage"):
mapping = {"mode":"wide", "rid":rid, "auto":use_auto}
st.components.v1.html(f"<script>setPreset('{ls_key}', '{json.dumps(mapping)}');</script>", height=0)
st.success("Saved preset.")


if 'long_df' in locals() and long_df is not None and not long_df.empty:
st.markdown("### Aggregated results")
agg = aggregate(long_df, agg_mode, formula)
st.dataframe(agg, use_container_width=True)


# Copy to clipboard (browser) via hidden textarea + JS
csv_text = agg.to_csv(index=False)
html_copy = f"""
<textarea id='data_csv' style='position:absolute;left:-9999px;'>""" + csv_text + """</textarea>
<button id='copybtn'>Copy table to clipboard</button>
<script>
document.getElementById('copybtn').onclick = function(){
const ta = document.getElementById('data_csv');
ta.select(); document.execCommand('copy');
alert('Copied!');
}
</script>
"""
st.components.v1.html(html_copy, height=40)


st.download_button("Download aggregated CSV", data=csv_text.encode("utf-8"), file_name="opportunity_aggregated.csv", mime="text/csv")


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
