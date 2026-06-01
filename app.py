import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
import ast
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Listings Explorer",
    page_icon="💼",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark slate background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] p {
    color: #e6edf3; 
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 800 !important; font-size: 1.8rem !important; }

/* Section headers */
h1 { font-weight: 800 !important; font-size: 2.4rem !important; letter-spacing: -0.03em; color: #e6edf3 !important; }
h2 { font-weight: 700 !important; color: #e6edf3 !important; }
h3 { font-weight: 600 !important; color: #8b949e !important; font-size: 0.82rem !important; letter-spacing: 0.1em; text-transform: uppercase; }

/* Divider */
hr { border-color: #21262d !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #21262d; }

/* Chart containers */
.chart-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 8px;
}

/* Caption */
.stCaption { color: #8b949e !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; }

/* Accent badge */
.accent-tag {
    display: inline-block;
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb55;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/*Tabs*/
.stTabs [data-baseweb="tab"] { color: #e6edf3; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ff4b4b; }
.stTabs [data-baseweb="tab"]:hover { color: #ff4b4b; }

</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(family="Syne, sans-serif", color="#8b949e", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    colorway=["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657", "#79c0ff"],
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1),
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "data/cleaned_jobs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "location"])
    df["company_name"] = df["company_name"].fillna("Unknown Company")
    df["normalized_salary"] = pd.to_numeric(df["normalized_salary"], errors="coerce")
    return df


# ── Load model & metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipe = joblib.load("model/salary_model.pkl")
    with open("model/salary_model_meta.json") as f:
        meta = json.load(f)
    return pipe, meta
    
df = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Explorer", 
    "🎯 Salary Predictor",
    "🧠 Skills Trends", 
    "📈 Market Trends", 
    "📝 Feedback"
])

with tab1:
    # ── Sidebar filters ────────────────────────────────────────────────────────────
    st.sidebar.markdown('<p style="color:#58a6ff;font-weight:800;font-size:1.1rem;letter-spacing:-0.01em;">🔍 Filters</p>', unsafe_allow_html=True)

    all_titles = sorted(df["title"].dropna().unique())
    selected_titles = st.sidebar.multiselect("Job Position", options=all_titles, placeholder="All positions")

    all_companies = sorted(df["company_name"].dropna().unique())
    selected_companies = st.sidebar.multiselect("Company", options=all_companies, placeholder="All companies")

    salary_df = df.dropna(subset=["normalized_salary"])
    sal_min = int(salary_df["normalized_salary"].min()) if not salary_df.empty else 0
    sal_max = int(salary_df["normalized_salary"].max()) if not salary_df.empty else 500_000

    salary_range = st.sidebar.slider("Annual Salary Range (USD)", min_value=sal_min, max_value=sal_max, value=(sal_min, sal_max), step=1_000, format="$%d")
    apply_salary_filter = st.sidebar.checkbox("Apply salary filter", value=False)

    all_locations = sorted(df["location"].dropna().unique())
    selected_locations = st.sidebar.multiselect("Location", options=all_locations, placeholder="All locations")

    remote_options = ["On-site", "Remote"]
    selected_remote = st.sidebar.multiselect("Work Type", options=remote_options, default=remote_options)

    st.sidebar.divider()
    st.sidebar.markdown('<p style="color:#58a6ff;font-weight:800;font-size:1.1rem;letter-spacing:-0.01em;">↕️ Sort Results</p>', unsafe_allow_html=True)
    sort_col = st.sidebar.selectbox("Sort by", options=["company_name", "remote_status", "normalized_salary", "location"],
        format_func=lambda x: {"company_name": "Company", "remote_status": "Remote Status", "normalized_salary": "Salary", "location": "Location"}[x])
    sort_asc = st.sidebar.radio("Order", ["Ascending", "Descending"]) == "Ascending"

    # ── Apply filters ──────────────────────────────────────────────────────────────
    filtered = df.copy()
    if selected_titles:
        filtered = filtered[filtered["title"].isin(selected_titles)]
    if selected_companies:
        filtered = filtered[filtered["company_name"].isin(selected_companies)]
    if apply_salary_filter:
        filtered = filtered[filtered["normalized_salary"].between(salary_range[0], salary_range[1])]
    if selected_locations:
        filtered = filtered[filtered["location"].isin(selected_locations)]
    if selected_remote:
        filtered = filtered[filtered["remote_status"].isin(selected_remote)]
    filtered = filtered.sort_values(by=sort_col, ascending=sort_asc, na_position="last")

    # ── Header ─────────────────────────────────────────────────────────────────────
    st.markdown('<div class="accent-tag">Live Dataset</div>', unsafe_allow_html=True)
    st.title("💼 Job Listings Explorer")
    st.caption("Filter and explore job postings interactively. Charts update with every filter.")

    # ── Summary metrics ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(filtered):,}")
    col2.metric("Unique Companies", f"{filtered['company_name'].nunique():,}")
    col3.metric("Unique Roles", f"{filtered['title'].nunique():,}")
    avg_sal = filtered["normalized_salary"].mean()
    col4.metric("Avg. Annual Salary", f"${avg_sal:,.0f}" if not pd.isna(avg_sal) else "N/A")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════════
    # CHART SECTION
    # ═══════════════════════════════════════════════════════════════════════════════

    def make_chart(fig):
        fig.update_layout(**PLOTLY_LAYOUT)
        return fig

    # ── Row 1: Position filter charts ─────────────────────────────────────────────
    st.markdown("### 📌 Job Position Insights")
    c1, c2 = st.columns(2)

    with c1:
        # Companies offering selected/all positions (top 15)
        pos_company = (
            filtered.groupby("company_name")["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "listings"})
            .sort_values("listings", ascending=True)
            .tail(15)
        )
        fig = go.Figure(go.Bar(
            x=pos_company["listings"],
            y=pos_company["company_name"],
            orientation="h",
            marker=dict(color="#58a6ff", opacity=0.85, line=dict(width=0)),
            text=pos_company["listings"],
            textposition="outside",
            textfont=dict(color="#8b949e", size=11),
        ))
        fig.update_layout(title=dict(text="Top Companies by Listing Count", font=dict(color="#e6edf3", size=14)))
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(make_chart(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        # Job positions distribution (top 12)
        title_counts = (
            filtered["title"].value_counts().head(12).reset_index()
        )
        title_counts.columns = ["title", "count"]
        fig2 = px.pie(
            title_counts, names="title", values="count",
            hole=0.55,
            color_discrete_sequence=["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff","#56d364","#ff7b72","#e3b341","#a5d6ff","#7ee787","#ffa198"],
        )
        fig2.update_traces(textposition="outside", textinfo="label+percent", textfont_size=11)
        fig2.update_layout(title=dict(text="Job Position Distribution", font=dict(color="#e6edf3", size=14)), showlegend=False)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(make_chart(fig2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Salary charts ───────────────────────────────────────────────────────
    st.markdown("### 💰 Salary Insights")
    c3, c4 = st.columns(2)

    with c3:
        # Companies by median salary (top 15)
        sal_by_company = (
            filtered.dropna(subset=["normalized_salary"])
            .groupby("company_name")["normalized_salary"]
            .median()
            .reset_index()
            .rename(columns={"normalized_salary": "median_salary"})
            .sort_values("median_salary", ascending=True)
            .tail(15)
        )
        fig3 = go.Figure(go.Bar(
            x=sal_by_company["median_salary"],
            y=sal_by_company["company_name"],
            orientation="h",
            marker=dict(color="#3fb950", opacity=0.85, line=dict(width=0)),
            text=sal_by_company["median_salary"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
            textfont=dict(color="#8b949e", size=11),
        ))
        fig3.update_layout(
            title=dict(text="Top Companies by Median Salary", font=dict(color="#e6edf3", size=14)),
            xaxis=dict(tickformat="$,.0f"),
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(make_chart(fig3), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        # Salary distribution histogram
        sal_data = filtered.dropna(subset=["normalized_salary"])
        fig4 = go.Figure(go.Histogram(
            x=sal_data["normalized_salary"],
            nbinsx=30,
            marker=dict(color="#d2a8ff", opacity=0.8, line=dict(width=0.5, color="#0d1117")),
        ))
        fig4.update_layout(
            title=dict(text="Salary Distribution", font=dict(color="#e6edf3", size=14)),
            xaxis=dict(title="Annual Salary (USD)", tickformat="$,.0f"),
            yaxis=dict(title="Number of Listings"),
            bargap=0.05,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(make_chart(fig4), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Row 3: Location & Remote charts ───────────────────────────────────────────
    st.markdown("### 🌍 Location & Work Type Insights")
    c5, c6 = st.columns(2)

    with c5:
        # Companies by location (top 15 locations by listing count)
        loc_counts = (
            filtered.groupby("location")["company_name"]
            .count()
            .reset_index()
            .rename(columns={"company_name": "listings"})
            .sort_values("listings", ascending=True)
            .tail(15)
        )
        fig5 = go.Figure(go.Bar(
            x=loc_counts["listings"],
            y=loc_counts["location"],
            orientation="h",
            marker=dict(color="#ffa657", opacity=0.85, line=dict(width=0)),
            text=loc_counts["listings"],
            textposition="outside",
            textfont=dict(color="#8b949e", size=11),
        ))
        fig5.update_layout(title=dict(text="Top Locations by Listing Count", font=dict(color="#e6edf3", size=14)))
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(make_chart(fig5), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        # Remote vs On-site breakdown per top companies
        remote_counts = (
            filtered.groupby(["company_name", "remote_status"])
            .size()
            .reset_index(name="count")
        )
        top_companies_remote = (
            remote_counts.groupby("company_name")["count"].sum()
            .nlargest(12).index.tolist()
        )
        remote_top = remote_counts[remote_counts["company_name"].isin(top_companies_remote)]

        fig6 = px.bar(
            remote_top,
            x="company_name",
            y="count",
            color="remote_status",
            barmode="stack",
            color_discrete_map={"Remote": "#58a6ff", "On-site": "#f78166", "Hybrid": "#3fb950"},
            text="count",
        )
        fig6.update_traces(textposition="inside", textfont_size=10)
        fig6.update_layout(
            title=dict(text="On-site vs Remote by Company", font=dict(color="#e6edf3", size=14)),
            xaxis=dict(title="", tickangle=-35),
            yaxis=dict(title="Listings"),
            legend=dict(title="Work Type"),
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(make_chart(fig6), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Row 4: Sort/Order chart ────────────────────────────────────────────────────
    st.markdown("### 📊 Sorted View — Company Overview")

    # Aggregate by company and show salary, remote_status, location in a bubble/scatter
    company_agg = (
        filtered.groupby("company_name")
        .agg(
            listings=("title", "count"),
            median_salary=("normalized_salary", "median"),
            locations=("location", "nunique"),
        )
        .reset_index()
        .dropna(subset=["median_salary"])
        .sort_values("listings", ascending=False)
        .head(40)
    )

    fig7 = px.scatter(
        company_agg,
        x="median_salary",
        y="listings",
        size="listings",
        color="locations",
        hover_name="company_name",
        color_continuous_scale=["#1f6feb", "#58a6ff", "#79c0ff"],
        labels={"median_salary": "Median Annual Salary (USD)", "listings": "Total Listings", "locations": "# Locations"},
        size_max=40,
    )
    fig7.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Company Landscape: Salary vs Listings (bubble = volume, color = locations)", font=dict(color="#e6edf3", size=14)),
    )
    fig7.update_layout(
        xaxis=dict(tickformat="$,.0f", gridcolor="#21262d"),
        height=420,
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Results table ──────────────────────────────────────────────────────────────
    st.markdown("### 📋 Full Listings Table")

    display_cols = {
        "company_name": "Company",
        "title": "Job Title",
        "location": "Location",
        "remote_status": "Work Type",
        "formatted_experience_level": "Experience Level",
        "normalized_salary": "Annual Salary (USD)",
        "work_type": "Employment Type",
    }

    display_df = (
        filtered[list(display_cols.keys())]
        .rename(columns=display_cols)
        .reset_index(drop=True)
    )
    display_df["Annual Salary (USD)"] = display_df["Annual Salary (USD)"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=520,
        column_config={
            "Posting URL": st.column_config.LinkColumn("Posting URL", display_text="View ↗"),
        },
    )
    st.caption(f"Showing {len(filtered):,} of {len(df):,} total listings.")

with tab2:

    # ─────────────────────────────────────────────────────────────────────────────
    from src.salary_predictor import render_salary_predictor
    render_salary_predictor(df)

with tab3:
    # ─────────────────────────────────────────────────────────────────────────────
    from src.skills_tab import render_skills_tab
    render_skills_tab(df)

with tab4:
    # ─────────────────────────────────────────────────────────────────────────────
    from src.trends_tab import render_trends_tab
    render_trends_tab(df)

with tab5:
    # ─────────────────────────────────────────────────────────────────────────────
    from src.usability_tab import render_usability_tab
    render_usability_tab()