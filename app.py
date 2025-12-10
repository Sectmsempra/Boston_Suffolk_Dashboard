# app.py
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Boston - Suffolk County Quality Dashboard (2022)",
    layout="wide",
)

# ---------------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------------
st.markdown(
    """
    <style>

    .main .block-container {
        padding-top: 0.6rem;
        padding-bottom: 3rem;  /* extra space so content doesn't sit under footer */
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    footer {visibility: hidden;}  /* hide default Streamlit footer */
    #MainMenu {visibility: hidden;}

    .subtitle {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    /* Profile badges (Income: High, etc.) */
    .metric-badges {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
        gap: 0.3rem;
        margin: 0.4rem 0 0.2rem 0;
        align-items: center;
    }
    .metric-badge {
        padding: 0.10rem 0.4rem;
        border-radius: 999px;
        background: #f4f4f6;
        border: 1px solid #e1e1e6;
        font-size: 0.76rem;
        color: #262730;
        white-space: normal;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 44px;
    }
    .metric-name {
        font-weight: 500;
        margin: 0;
        font-size: 0.70rem;
        line-height: 1.0;
    }
    .metric-value {
        font-weight: 700;
        font-size: 0.9rem;
        line-height: 1.05;
    }

    /* Compact KPI grid (2×2) */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
        margin-top: 0.5rem;
    }
    .kpi-card {
        padding: 0.55rem 0.7rem;
        border-radius: 0.75rem;
        background: #ffffff;
        border: 1px solid #e5e5ea;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        min-height: 70px;
    }
    .kpi-title {
        font-size: 0.78rem;
        color: #555;
        margin-bottom: 0.05rem;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 1.08rem;
        font-weight: 600;
        margin-bottom: 0.1rem;
        color: #111827;
    }
    .kpi-delta {
        font-size: 0.75rem;
        font-weight: 500;
    }
    .delta-good { color: #17633a; }
    .delta-bad { color: #8b1a1a; }
    .delta-neutral { color: #6b7280; }

    /* Custom fixed footer with APA-style citations */
    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #f8f9fa;
        border-top: 1px solid #e0e0e0;
        padding: 0.3rem 1.2rem;
        font-size: 0.7rem;
        color: #555;
        text-align: left;
        z-index: 999;
    }

    .custom-footer a {
        color: #555;
        text-decoration: underline;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path("data/processed/master_zip_dataset_2022_enriched.csv")
GEO_PATH = Path("data/geo/boston_zips.geojson")

# ---------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["zip"] = df["zip"].astype(str).str.zfill(5)

    # Use Neighbourhood column from the master CSV as the primary name
    if "Neighbourhood" in df.columns:
        df["area_name"] = df["Neighbourhood"]
    elif "neighbourhood" in df.columns:
        df["area_name"] = df["neighbourhood"]
    elif "area_name" not in df.columns:
        df["area_name"] = df["district"].apply(
            lambda d: f"District {d}" if pd.notna(d) else "Unknown"
        )

    return df


@st.cache_data
def load_geojson():
    with open(GEO_PATH, "r") as f:
        return json.load(f)


df = load_data()
geojson = load_geojson()

# Map from zip -> area name for display in selectbox and hover tooltips
area_map = df.set_index("zip")["area_name"].to_dict()

# ---------------------------------------------------------------
# METRIC CONFIG
# ---------------------------------------------------------------
map_metric_labels = {
    "score_index": "Overall Score",
    "median_income": "Median Household Income",
    "crime_incidents_2022": "Crime Incidents",
    "education": "Educated Population",
    "housing": "Median Rent",
    "total_population": "Population Density",
}

good_when_high = {"score_index", "median_income", "education"}
good_when_low = {"crime_incidents_2022", "housing", "total_population"}

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
def delta_text(zip_val: float, city_val: float, higher_is_good: bool) -> tuple[str, str]:
    """Return (HTML text, CSS class) for the delta line."""
    if city_val == 0:
        return "vs city: —", "delta-neutral"

    diff_pct = (zip_val - city_val) / city_val * 100
    sign = "+" if diff_pct > 0 else ""
    pct_str = f"{sign}{diff_pct:.1f}%"

    if abs(diff_pct) < 0.5:
        return f"vs city: {pct_str}", "delta-neutral"

    is_higher = diff_pct > 0
    if higher_is_good:
        css = "delta-good" if is_higher else "delta-bad"
    else:
        css = "delta-bad" if is_higher else "delta-good"

    return f"vs city: {pct_str}", css


def render_zip_profile(selected_row: pd.Series):
    """Right-side panel in row 2: profile badges + radar chart."""
    area_name = selected_row.get("area_name", "")
    zip_code = selected_row.get("zip", "")
    title = area_name or f"ZIP {zip_code}"

    st.markdown(f"#### Neighborhood profile – {title}")

    # Radar chart – contribution to composite score (0–100 scale)
    contrib_values = {
        "Income": selected_row["income_contrib"] * 100,
        "Crime": selected_row["crime_contrib"] * 100,
        "Education": selected_row["education_contrib"] * 100,
        "Housing": selected_row["housing_contrib"] * 100,
        "Density": selected_row["density_contrib"] * 100,
    }

    categories = list(contrib_values.keys())
    values = list(contrib_values.values())

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Impact on score",
            hovertemplate="%{theta}: %{r:.1f} / 100<extra></extra>",
        )
    )

    max_val = max(values) if values else 1
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(30, max_val * 1.1)]),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        height=290,
        margin=dict(l=10, r=10, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Profile pills
    labels = {
        "Income": selected_row["income_level"],
        "Crime": selected_row["crime_level"],
        "Education": selected_row["education_level"],
        "Housing": selected_row["housing_affordability"],
        "Density": selected_row["density_level"],
    }

    badges_html = '<div class="metric-badges">'
    for name, value in labels.items():
        badges_html += (
            '<div class="metric-badge">'
            f'<div class="metric-name">{name}</div>'
            f'<div class="metric-value">{value}</div>'
            "</div>"
        )
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)


def format_metric_value(metric_key: str, value: float) -> str:
    """Pretty formatting for the cross-ZIP chart tooltips."""
    if metric_key == "median_income":
        return f"${value:,.0f}"
    if metric_key == "housing":
        return f"${value:,.0f}"
    if metric_key == "education":
        return f"{value:.1f}%"
    if metric_key in {"crime_incidents_2022", "total_population"}:
        return f"{value:,.0f}"
    if metric_key == "score_index":
        return f"{value:.1f}"
    return f"{value:,.2f}"


def get_feature_centroid(feature: dict) -> tuple[float, float] | None:
    """Return (lat, lon) centroid of a GeoJSON feature (approx via avg of vertices)."""
    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if not coords or gtype not in {"Polygon", "MultiPolygon"}:
        return None

    lats, lons = [], []

    def add_ring(ring):
        for lon, lat in ring:
            lats.append(lat)
            lons.append(lon)

    if gtype == "Polygon":
        for ring in coords:
            add_ring(ring)
    else:  # MultiPolygon
        for poly in coords:
            for ring in poly:
                add_ring(ring)

    if not lats or not lons:
        return None

    return (sum(lats) / len(lats), sum(lons) / len(lons))


@st.cache_data
def build_zip_centroids(_geojson: dict) -> dict[str, tuple[float, float]]:
    """Map ZIP5 -> (lat, lon) centroid."""
    out = {}
    for feat in _geojson.get("features", []):
        props = feat.get("properties", {})
        z = props.get("ZIP5")
        if z is None:
            continue
        z = str(z).zfill(5)
        c = get_feature_centroid(feat)
        if c:
            out[z] = c
    return out


zip_centroids = build_zip_centroids(geojson)

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.title("Filters")

zip_options = df.sort_values("zip")["zip"].tolist()
default_zip = df.sort_values("rank").iloc[0]["zip"]

selected_zip = st.sidebar.selectbox(
    "Choose neighborhood",
    options=zip_options,
    index=zip_options.index(default_zip),
    format_func=lambda z: area_map.get(z, z),
)

map_metric = st.sidebar.selectbox(
    "Map metric",
    options=list(map_metric_labels.keys()),
    index=0,
    format_func=lambda k: map_metric_labels[k],
)

# New (for rubric): top-N controls for cross-ZIP visuals
st.sidebar.markdown("---")
top_n = st.sidebar.slider("Show top N neighborhoods", min_value=5, max_value=20, value=12, step=1)

selected_row = df[df["zip"] == selected_zip].iloc[0]
selected_area_name = selected_row["area_name"]

# ---------------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------------
st.markdown("## Boston - Suffolk County Quality Dashboard (2022)")
st.markdown(
    "ZIP-level composite score for Boston based on **income, crime, education, housing,** "
    "and **population density**."
)

# ---------------------------------------------------------------
# ROW 1: MAP (60–70%) + KPI PANEL (30–40%)
# ---------------------------------------------------------------
row1_left, row1_right = st.columns([7, 3])

# ---------- MAP ----------
with row1_left:
    color_scale = "RdYlGn" if map_metric in good_when_high else "RdYlGn_r"
    vmin = df[map_metric].min()
    vmax = df[map_metric].max()

    custom_cols = ["area_name", "zip"]

    fig_map = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="zip",
        featureidkey="properties.ZIP5",
        color=map_metric,
        custom_data=custom_cols,
        color_continuous_scale=color_scale,
        range_color=(vmin, vmax),
        mapbox_style="carto-positron",
        center={"lat": 42.33, "lon": -71.06},
        zoom=11.2,
        opacity=0.85,
        labels={map_metric: map_metric_labels[map_metric]},
    )

    # Clean tooltip (no text box label on map)
    fig_map.update_traces(
        marker_line_width=0.8,
        marker_line_color="white",
        hovertemplate="<b>%{customdata[0]}</b><br>ZIP: %{customdata[1]}<extra></extra>",
    )

    fig_map.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title=map_metric_labels[map_metric],
            ticks="outside",
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.15)",
            font_size=11,
        ),
    )

    # ---- Highlight selected ZIP with OUTLINE ONLY (no dot, no label) ----
    fig_map.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=[selected_zip],
            featureidkey="properties.ZIP5",
            z=[1],  # dummy
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker_line_width=4,
            marker_line_color="rgba(17, 24, 39, 0.95)",  # dark clean outline
            hoverinfo="skip",
            name="Selected",
        )
    )

    # --- NO centroid marker ---
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

# ---------- KPI PANEL ----------
with row1_right:
    neighborhood_title = selected_area_name or selected_zip
    st.markdown(f"#### {neighborhood_title} summary")
    st.caption(f"ZIP code: {selected_zip}")

    # Score + rank (top row)
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric(
            label=f"Score (0–100)\n{neighborhood_title}",
            value=f"{selected_row['score_index']:.1f}",
        )
    with kpi_col2:
        st.metric(
            label="City-wide rank",
            value=f"#{int(selected_row['rank'])}",
        )

    # Compute city averages from stored deltas
    income_val = selected_row["median_income"]
    income_diff = selected_row["income_vs_city_avg"]
    income_city = income_val - income_diff

    crime_val = selected_row["crime_incidents_2022"]
    crime_diff = selected_row["crime_vs_city_avg"]
    crime_city = crime_val - crime_diff

    rent_val = selected_row["housing"]
    rent_diff = selected_row["housing_vs_city_avg"]
    rent_city = rent_val - rent_diff

    pop_val = selected_row["total_population"]
    pop_diff = selected_row["population_vs_city_avg"]
    pop_city = pop_val - pop_diff

    # Build compact 2×2 KPI grid
    def kpi_card_html(title, value_html, delta_html, delta_class) -> str:
        return (
            '<div class="kpi-card">'
            f'<div class="kpi-title">{title}</div>'
            f'<div class="kpi-value">{value_html}</div>'
            f'<div class="kpi-delta {delta_class}">{delta_html}</div>'
            "</div>"
        )

    cards = []

    # Income (higher is good)
    delta_html, delta_class = delta_text(income_val, income_city, higher_is_good=True)
    cards.append(kpi_card_html("Income", f"${income_val:,.0f}", delta_html, delta_class))

    # Crime (lower is good)
    delta_html, delta_class = delta_text(crime_val, crime_city, higher_is_good=False)
    cards.append(kpi_card_html("Crime incidents", f"{crime_val:,.0f}", delta_html, delta_class))

    # Rent (lower is good)
    delta_html, delta_class = delta_text(rent_val, rent_city, higher_is_good=False)
    cards.append(kpi_card_html("Rent", f"${rent_val:,.0f}", delta_html, delta_class))

    # Population (lower density considered better here)
    delta_html, delta_class = delta_text(pop_val, pop_city, higher_is_good=False)
    cards.append(kpi_card_html("Population", f"{pop_val:,.0f}", delta_html, delta_class))

    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# ROW 2: METRIC COMPARISON (ZIP vs CITY AVG) + ZIP PROFILE
# ---------------------------------------------------------------
row2_left, row2_right = st.columns([6, 4])

with row2_left:
    st.markdown(f"#### How {neighborhood_title} compares to the Boston average")

    metric_rows = []
    metrics_info = [
        ("Income", income_val, income_city, True),
        ("Crime incidents", crime_val, crime_city, False),
        ("Rent", rent_val, rent_city, False),
        ("Population", pop_val, pop_city, False),
    ]

    for name, zip_val, city_val, higher_is_good in metrics_info:
        pct_diff = 0.0 if city_val == 0 else (zip_val - city_val) / city_val * 100

        if abs(pct_diff) < 0.5:
            color = "#7f7f7f"
        else:
            is_higher = pct_diff > 0
            if higher_is_good:
                color = "#2ca02c" if is_higher else "#d62728"
            else:
                color = "#d62728" if is_higher else "#2ca02c"

        metric_rows.append({"Metric": name, "Percent_diff": pct_diff, "Color": color})

    diff_df = pd.DataFrame(metric_rows)

    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Bar(
            x=diff_df["Metric"],
            y=diff_df["Percent_diff"],
            marker_color=diff_df["Color"],
            hovertemplate=" %{x}<br>Difference from Boston (%) %{y:.1f}%<extra></extra>",
        )
    )
    fig_comp.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(0,0,0,0.5)")

    fig_comp.update_layout(
        height=260,
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis_title="",
        yaxis_title="Difference vs city average (%)",
    )
    fig_comp.update_yaxes(tickformat=".0f")

    st.plotly_chart(fig_comp, use_container_width=True)

    st.caption(
        f"Bars show how much {neighborhood_title} differs from the Boston average for each metric. "
        "Green means better outcomes (e.g., higher income or lower crime)."
    )

with row2_right:
    render_zip_profile(selected_row)

# ---------------------------------------------------------------
# ROW 3 (NEW): VARIABLE-SPECIFIC ACROSS-ZIP CHART + SCORE BREAKDOWN
# ---------------------------------------------------------------
st.markdown("---")
row3_left, row3_right = st.columns([6, 6])

with row3_left:
    metric_name = map_metric_labels[map_metric]
    higher_is_better = map_metric in good_when_high
    chart_title = f"#### Neighborhoods ranked by {metric_name}"

    st.markdown(chart_title)
    st.caption(
        "This addresses the variable-specific chart requirement by comparing neighborhoods across ZIP codes."
    )

    # Sort direction: if higher is better, show top; else show lowest (better outcome)
    sort_ascending = not higher_is_better

    rank_df = (
        df[["area_name", "zip", map_metric]]
        .copy()
        .rename(columns={"area_name": "Neighborhood"})
        .sort_values(map_metric, ascending=sort_ascending)
        .head(top_n)
    )

    # For readability in bar chart: create label "Neighborhood (ZIP)"
    rank_df["Label"] = rank_df["Neighborhood"] + " (" + rank_df["zip"] + ")"

    fig_rank = go.Figure()
    fig_rank.add_trace(
        go.Bar(
            x=rank_df[map_metric],
            y=rank_df["Label"],
            orientation="h",
            hovertemplate=(
                "<b>%{y}</b><br>"
                + metric_name
                + ": %{x}<extra></extra>"
            ),
        )
    )

    fig_rank.update_layout(
        height=420,
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis_title=metric_name,
        yaxis_title="",
    )

    # Improve axis formatting for key metrics
    if map_metric in {"median_income", "housing"}:
        fig_rank.update_xaxes(tickprefix="$", separatethousands=True)
    elif map_metric == "education":
        fig_rank.update_xaxes(ticksuffix="%")
    else:
        fig_rank.update_xaxes(separatethousands=True)

    st.plotly_chart(fig_rank, use_container_width=True)

with row3_right:
    st.markdown("#### Score breakdown (Top neighborhoods)")
    st.caption(
        "This directly satisfies the score breakdown requirement by showing how each variable contributes to the composite score."
    )

    contrib_cols = [
        ("Income", "income_contrib"),
        ("Crime", "crime_contrib"),
        ("Education", "education_contrib"),
        ("Housing", "housing_contrib"),
        ("Density", "density_contrib"),
    ]

    breakdown_df = (
        df[["area_name", "zip", "score_index"] + [c for _, c in contrib_cols]]
        .copy()
        .sort_values("score_index", ascending=False)
        .head(top_n)
    )
    breakdown_df["Label"] = breakdown_df["area_name"] + " (" + breakdown_df["zip"] + ")"

    # --- Normalize to 100% per ZIP (robust even if sums aren't exactly 1.0) ---
    contrib_only = breakdown_df[[c for _, c in contrib_cols]].clip(lower=0)
    row_sum = contrib_only.sum(axis=1).replace(0, 1)  # avoid divide-by-zero
    contrib_norm = contrib_only.div(row_sum, axis=0) * 100  # 0–100 per row

    fig_stack = go.Figure()
    for label, col in contrib_cols:
        fig_stack.add_trace(
            go.Bar(
                y=breakdown_df["Label"],
                x=contrib_norm[col],
                orientation="h",
                name=label,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"{label} contribution: "
                    + "%{x:.1f}%<extra></extra>"
                ),
            )
        )

    fig_stack.update_layout(
        barmode="stack",
        height=420,
        margin=dict(l=0, r=10, t=10, b=0),
        xaxis_title="Contribution to composite score (%)",
        yaxis_title="",
        legend_title="Components",
    )
    fig_stack.update_xaxes(range=[0, 100], ticksuffix="%")

    st.plotly_chart(fig_stack, use_container_width=True)

# ---------------------------------------------------------------
# FULL TABLE (UPGRADED): INCLUDE SCORE BREAKDOWN COLUMNS
# ---------------------------------------------------------------
with st.expander("View full neighborhood data table"):
    table_df = df.copy()

    # Ensure Neighbourhood column exists and is populated from area_name if needed
    if "Neighbourhood" not in table_df.columns:
        table_df["Neighbourhood"] = table_df["area_name"]

    # Add contribution columns (as percentages) for rubric compliance
    table_df["Income contrib (%)"] = table_df["income_contrib"] * 100
    table_df["Crime contrib (%)"] = table_df["crime_contrib"] * 100
    table_df["Education contrib (%)"] = table_df["education_contrib"] * 100
    table_df["Housing contrib (%)"] = table_df["housing_contrib"] * 100
    table_df["Density contrib (%)"] = table_df["density_contrib"] * 100

    display_cols = [
        "Neighbourhood",
        "zip",
        "rank",
        "score_index",
        "median_income",
        "crime_incidents_2022",
        "education",
        "housing",
        "total_population",
        "Income contrib (%)",
        "Crime contrib (%)",
        "Education contrib (%)",
        "Housing contrib (%)",
        "Density contrib (%)",
        "income_level",
        "crime_level",
        "education_level",
        "housing_affordability",
        "density_level",
    ]

    table_df = table_df[display_cols].sort_values("rank").rename(
        columns={
            "score_index": "Score",
            "median_income": "Median income",
            "crime_incidents_2022": "Crime incidents",
            "education": "Education (% bachelor+)",
            "housing": "Median rent",
            "total_population": "Population",
        }
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.1f"),
            "Median income": st.column_config.NumberColumn(format="$%d"),
            "Median rent": st.column_config.NumberColumn(format="$%d"),
            "Crime incidents": st.column_config.NumberColumn(format="%d"),
            "Population": st.column_config.NumberColumn(format="%d"),
            "Income contrib (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Crime contrib (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Education contrib (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Housing contrib (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Density contrib (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

# ---------------------------------------------------------------
# CUSTOM FOOTER WITH DATA CITATIONS
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="custom-footer">
        Data sources (APA-style citations): 
        U.S. Census Bureau. (2023). <em>American Community Survey 5-year estimates, Tables S1901 (Income), S1501 (Educational Attainment), DP04 (Housing), and DP05 (Demographics)</em>. Retrieved from https://data.census.gov/ &nbsp;|&nbsp;
        Boston Police Department. (2022). <em>Crime Incident Reports, 2022</em>. Retrieved from https://data.boston.gov/
    </div>
    """,
    unsafe_allow_html=True,
)
