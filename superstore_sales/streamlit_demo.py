# app.py — Streamlit + Plotly dashboard starter
# -------------------------------------------------
# Features
# - CSV upload (or auto-loads sample dataset if no file)
# - Smart dtype parsing (dates, categoricals, numerics)
# - Sidebar filters (date range, categorical multi-selects)
# - Simple aggregations (sum/mean/count) by a chosen dimension
# - KPIs, interactive Plotly charts, data preview
# - Download filtered/aggregated data as CSV
# -------------------------------------------------

import io
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------
# Page & style
# ----------------------------
st.set_page_config(
    page_title="Pandas → Interactive Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Pandas → Interactive Dashboard (Streamlit + Plotly)")
st.caption(
    "Upload your CSV on the left, or explore with the built‑in sample. "
    "Use the sidebar to filter, group, and visualize."
)

# ----------------------------
# Helpers
# ----------------------------

def _try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to parse likely date columns in place."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            # Heuristic: if vast majority of values look like dates, parse.
            # To do: add more possible regex patterns for dates
            date_hits = sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}", regex=True).mean()
            if date_hits > 0.9:
                df[col] = pd.to_datetime(df[col], errors="ignore")
    return df

@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    # Plotly's sample: gapminder
    df = px.data.gapminder().rename(columns={"year": "Year", "continent": "Continent", "country": "Country"})
    # Create a pseudo date from Year for demo purposes
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
    return df

@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = _try_parse_dates(df)
    return df

# ----------------------------
# Sidebar: data source
# ----------------------------
st.sidebar.header("1) Data source")
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    df = load_csv(file)
    source_label = "Your CSV"
else:
    df = load_sample()
    source_label = "Sample: Plotly Gapminder"

st.write(f"**Data source:** {source_label} — {len(df):,} rows × {len(df.columns)} columns")

# Identify columns by type
date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]

# ----------------------------
# Sidebar: filters
# ----------------------------
st.sidebar.header("2) Filters")

# Date range filter (optional)
if date_cols:
    date_col = st.sidebar.selectbox("Date column", options=date_cols, index=0)
    min_date, max_date = pd.to_datetime(df[date_col].min()), pd.to_datetime(df[date_col].max())
    start, end = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(start, tuple):
        # Streamlit may return a tuple on first render; normalize
        start, end = start
    mask_date = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
else:
    date_col = None
    mask_date = pd.Series(True, index=df.index)

# Categorical filters (multi-select per column)
active_cat_filters = {}
for c in cat_cols[:6]:  # limit to first 6 categoricals in UI to avoid clutter
    unique_vals = df.loc[mask_date, c].dropna().unique()
    if 1 < len(unique_vals) <= 200:
        choices = st.sidebar.multiselect(f"Filter: {c}", options=sorted(map(str, unique_vals)))
        if choices:
            active_cat_filters[c] = set(choices)

mask_cat = pd.Series(True, index=df.index)
for c, allowed in active_cat_filters.items():
    mask_cat &= df[c].astype(str).isin(allowed)

filtered = df.loc[mask_date & mask_cat].copy()

# ----------------------------
# KPIs
# ----------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Rows (filtered)", f"{len(filtered):,}")
with kpi_cols[1]:
    st.metric("Columns", f"{len(filtered.columns):,}")
with kpi_cols[2]:
    pass
    # if date_col:
     #   st.metric("From", pd.to_datetime(filtered[date_col].min()).date() if not filtered.empty else "—")
with kpi_cols[3]:
    pass
    # if date_col:
    #     st.metric("To", pd.to_datetime(filtered[date_col].max()).date() if not filtered.empty else "—")

st.divider()

# ----------------------------
# Sidebar: aggregation & chart config
# ----------------------------
st.sidebar.header("3) Aggregate & visualize")
dim = st.sidebar.selectbox("Group by (dimension)", options=(cat_cols + date_cols) if (cat_cols or date_cols) else [None])
metric = st.sidebar.selectbox("Metric (numeric)", options=num_cols if num_cols else [None])
aggr = st.sidebar.selectbox("Aggregation", options=["sum", "mean", "count"])
chart_type = st.sidebar.selectbox("Chart type", options=["Line", "Bar", "Scatter", "Box", "Area"])

# Compute aggregation
if dim and metric and not filtered.empty:
    if aggr == "count":
        agg_df = filtered.groupby(dim, dropna=False).size().reset_index(name="value")
        value_col = "value"
    else:
        agg_df = (
            filtered.groupby(dim, dropna=False)[metric]
            .agg(aggr)
            .reset_index(name=f"{aggr}_{metric}")
        )
        value_col = f"{aggr}_{metric}"
else:
    agg_df = pd.DataFrame()
    value_col = None

# ----------------------------
# Layout: left (table) | right (chart)
# ----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Data preview")
    st.dataframe(filtered.head(1000), use_container_width=True)

    # Download filtered & aggregated data
    st.markdown("**Download**")
    if not filtered.empty:
        st.download_button(
            "⬇️ Filtered data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv",
        )
    if not agg_df.empty:
        st.download_button(
            "⬇️ Aggregated data (CSV)",
            data=agg_df.to_csv(index=False).encode("utf-8"),
            file_name="aggregated_data.csv",
            mime="text/csv",
        )

with right:
    st.subheader("Interactive chart")
    if agg_df.empty or value_col is None:
        st.info("Select a *dimension*, *metric*, and *aggregation* in the sidebar to render a chart.")
    else:
        # Pick a suitable Plotly Express chart based on user selection
        if chart_type == "Line":
            fig = px.line(agg_df, x=dim, y=value_col, markers=True)
        elif chart_type == "Bar":
            fig = px.bar(agg_df, x=dim, y=value_col)
        elif chart_type == "Scatter":
            fig = px.scatter(agg_df, x=dim, y=value_col)
        elif chart_type == "Box":
            # Box plots need raw distribution; fallback to raw metric if possible
            if metric:
                fig = px.box(filtered, x=dim, y=metric)
            else:
                fig = px.box(agg_df, y=value_col)
        elif chart_type == "Area":
            fig = px.area(agg_df, x=dim, y=value_col)
        else:
            fig = px.line(agg_df, x=dim, y=value_col)

        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("How this works (quick tour)"):
    st.markdown(
        """
        **Streamlit** builds the app UI from this Python script. **Pandas** handles data wrangling. **Plotly** renders interactive charts.

        1. **Load data**: Upload a CSV or use the bundled Gapminder sample. We try to parse date-like columns.
        2. **Filter**: Date range filter (if a date column exists) + categorical multi-selects (first few categoricals).
        3. **Aggregate**: Choose a *dimension* (group-by column), *metric* (numeric), and an aggregation (sum/mean/count).
        4. **Visualize**: Plotly Express draws the chart you pick (line/bar/scatter/box/area).
        5. **Export**: Download filtered or aggregated results as CSV.

        ✨ Tip: Replace the sample with your own dataset and tweak: default filters, calculated columns, custom KPIs, and multi‑page nav via `st.page_link`.
        """
    )

st.caption("Built with ❤️ using Streamlit, Plotly, and Pandas.")
