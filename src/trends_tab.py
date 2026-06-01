"""
trends_tab.py — Google Trends vs Job Listings tab
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.graph_objects as go
import plotly.express as px

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Syne, sans-serif", color="#8b949e", size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1),
)

SKILL_COLORS = [
    "#58a6ff","#3fb950","#f78166","#d2a8ff",
    "#ffa657","#79c0ff","#56d364","#ff7b72",
    "#e3b341","#a5d6ff","#7ee787","#ffa198",
    "#ffb77c","#c3d6f5","#b3f0c0",
]

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_trends():
    try:
        skills  = pd.read_csv("data/trends_combined.csv", parse_dates=["date"])
        return skills
    except FileNotFoundError:
        return None

@st.cache_data
def load_title_trends():
    try:
        return pd.read_csv("data/trends_jobtitles_long.csv", parse_dates=["date"])
    except FileNotFoundError:
        return None

@st.cache_data
def load_skill_listings():
    try:
        return pd.read_csv("data/skill_weekly_listings.csv", parse_dates=["week"])
    except FileNotFoundError:
        return None

def parse_list(val):
    if isinstance(val, list): return val
    if pd.isna(val): return []
    try: return ast.literal_eval(str(val))
    except: return []


def render_trends_tab(df: pd.DataFrame):
    st.markdown('<div class="accent-tag">Dataset 2 · Google Trends via pytrends</div>', unsafe_allow_html=True)
    st.title("📈 Market Trends")
    st.caption(
        "Google Search interest (via pytrends web scraping) combined with "
        "actual job listing volume from LinkedIn. "
        "Reveals whether public demand for skills tracks what companies post."
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    trends      = load_trends()
    title_trends = load_title_trends()
    skill_vols  = load_skill_listings()

    if trends is None:
        st.error(
            "⚠️ `data/trends_combined.csv` not found.  \n"
            "Run `google_trends.ipynb` on your **local machine** first.  \n"
            "(`pytrends` requires a non-datacenter IP — run it from your laptop, not a cloud server.)"
        )
        # Show what the tab will look like once data is present
        st.info(
            "**Once you run the notebook, this tab will show:**\n"
            "- Google search interest trends for top skills (2023–2024)\n"
            "- Comparison: search interest vs actual job listing volume\n"
            "- Skills with rising vs declining public interest\n"
            "- Skill gap analysis: high search interest but low job postings\n"
            "- Job title search trends (software engineer vs data scientist etc.)"
        )
        st.stop()

    # ── About card ────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About this data", expanded=False):
        skills_available = sorted(trends["skill"].unique())
        c1, c2, c3 = st.columns(3)
        c1.metric("Data Source",   "Google Trends (pytrends)")
        c2.metric("Method",        "Web scraping")
        c3.metric("Skills Tracked", len(skills_available))
        st.caption(
            "**pytrends** is an unofficial Google Trends API wrapper. "
            "Interest values are relative (0–100), where 100 = peak popularity "
            "in the selected period. Values are weekly, geo=US. "
            "Batched queries use a common anchor keyword for cross-batch normalisation."
        )
        st.caption(f"Skills tracked: {', '.join(skills_available)}")

    st.divider()

    # ── 1. Skill search interest over time ─────────────────────────────────────
    st.markdown("### 🔍 Google Search Interest — Skills Over Time")
    st.caption("Higher = more people searching for this skill on Google that week.")

    all_skills = sorted(trends["skill"].unique())
    default_skills = all_skills[:5] if len(all_skills) >= 5 else all_skills

    sel_skills = st.multiselect(
        "Select skills to compare",
        options=all_skills,
        default=default_skills,
        key="trends_skills_select",
    )

    if sel_skills:
        filt_trends = trends[trends["skill"].isin(sel_skills)]

        fig1 = go.Figure()
        for i, skill in enumerate(sel_skills):
            d = filt_trends[filt_trends["skill"] == skill].sort_values("date")
            color = SKILL_COLORS[i % len(SKILL_COLORS)]
            # Smoothed line
            smoothed = d["interest"].rolling(3, center=True, min_periods=1).mean()
            fig1.add_trace(go.Scatter(
                x=d["date"], y=smoothed,
                mode="lines",
                name=skill.title(),
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{skill.title()}</b><br>Week: %{{x|%b %d, %Y}}<br>Interest: %{{y:.0f}}<extra></extra>",
            ))
            # Raw values as faint dots
            fig1.add_trace(go.Scatter(
                x=d["date"], y=d["interest"],
                mode="markers",
                name=f"{skill} (raw)",
                marker=dict(color=color, size=3, opacity=0.3),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig1.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Google Search Interest (weekly, US)", font=dict(color="#e6edf3", size=14)),
        )
        fig1.update_layout(
            height=400,
            xaxis=dict(title=""),
            yaxis=dict(title="Interest (0–100)", range=[0, 105]),
            hovermode="x unified",
        )
        
        st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ── 2. Average interest ranking ────────────────────────────────────────────
    st.markdown("### 🏆 Average Search Interest Ranking")
    st.caption("Which skills people search for most consistently over the full period.")

    avg_interest = (
        trends.groupby("skill")["interest"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    avg_interest["color"] = [
        SKILL_COLORS[i % len(SKILL_COLORS)]
        for i in range(len(avg_interest))
    ]

    fig2 = go.Figure(go.Bar(
        x=avg_interest["interest"],
        y=avg_interest["skill"].str.title(),
        orientation="h",
        marker=dict(color=avg_interest["color"], line=dict(width=0)),
        text=avg_interest["interest"].round(1),
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Average Google Search Interest by Skill (full period)",
                   font=dict(color="#e6edf3", size=14)),
    )
    fig2.update_layout(
        height=max(300, len(avg_interest) * 32),
        xaxis=dict(title="Average Interest (0–100)", range=[0, 115]),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 3. Search interest vs listing volume ───────────────────────────────────
    st.markdown("### ⚡ Search Interest vs Job Listing Volume")

    if skill_vols is None or skill_vols.empty:
        st.info(
            "Listing volume data (`data/skill_weekly_listings.csv`) not found.  \n"
            "Re-run `finals.ipynb` to add time features, then re-run `google_trends.ipynb`."
        )
    else:
        st.caption(
            "Left axis: Google search interest (0–100).  "
            "Right axis: number of job listings mentioning the skill that week.  "
            "A strong correlation means public interest and hiring move together."
        )

        compare_skill = st.selectbox(
            "Select skill to compare",
            options=sorted(trends["skill"].unique()),
            key="compare_skill_select",
        )

        trend_d = (
            trends[trends["skill"] == compare_skill]
            .sort_values("date")
            .set_index("date")["interest"]
        )
        listing_d = (
            skill_vols[skill_vols["skill"] == compare_skill]
            .sort_values("week")
            .set_index("week")["listing_count"]
        )

        # Align on shared dates
        shared_idx = trend_d.index.intersection(listing_d.index)

        if len(shared_idx) < 3:
            st.warning(f"Not enough overlapping dates for '{compare_skill}'. Try another skill.")
        else:
            fig3 = go.Figure()
            # Google interest
            fig3.add_trace(go.Scatter(
                x=trend_d.index, y=trend_d.values,
                name="Google Search Interest",
                line=dict(color="#58a6ff", width=2.5),
                yaxis="y1",
            ))
            # Listing volume
            fig3.add_trace(go.Bar(
                x=listing_d.index, y=listing_d.values,
                name="Job Listing Count",
                marker=dict(color="#3fb950", opacity=0.5),
                yaxis="y2",
            ))

            corr = trend_d.loc[shared_idx].corr(listing_d.loc[shared_idx])

            fig3.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text=f'"{compare_skill.title()}" — Search Interest vs Listing Volume  (r = {corr:.2f})',
                    font=dict(color="#e6edf3", size=13),
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
            )
            fig3.update_layout(
                height=400,
                yaxis=dict(title="Google Interest (0–100)", side="left"),
                yaxis2=dict(title="Job Listings", side="right", overlaying="y",
                            showgrid=False),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Correlation metric
            m1, m2, m3 = st.columns(3)
            m1.metric("Correlation (r)", f"{corr:.3f}")
            m2.metric("Avg Search Interest", f"{trend_d.mean():.1f} / 100")
            m3.metric("Avg Weekly Listings", f"{listing_d.mean():.0f}")

            if abs(corr) > 0.6:
                st.success(f"Strong {'positive' if corr > 0 else 'negative'} correlation — Google search interest and job postings move together for **{compare_skill}**.")
            elif abs(corr) > 0.3:
                st.info(f"Moderate correlation — some relationship between search interest and listings for **{compare_skill}**.")
            else:
                st.warning(f"Weak correlation — search interest and listing volume appear independent for **{compare_skill}**.")

    st.divider()

    # ── 4. Skill gap analysis ──────────────────────────────────────────────────
    st.markdown("### 🎯 Skill Gap Analysis")
    st.caption(
        "Skills where **search interest is high but listing volume is low** = "
        "skills people want to learn but companies aren't hiring for yet (emerging skills).  \n"
        "Skills where **listing volume is high but search interest is low** = "
        "skills companies need but fewer people are searching for (opportunity skills)."
    )

    if skill_vols is not None and not skill_vols.empty:
        avg_trend   = trends.groupby("skill")["interest"].mean()
        avg_listings = skill_vols.groupby("skill")["listing_count"].mean()

        gap_df = pd.DataFrame({
            "avg_interest":  avg_trend,
            "avg_listings":  avg_listings,
        }).dropna().reset_index()
        gap_df.columns = ["skill", "avg_interest", "avg_listings"]

        # Normalise both to 0-1 scale
        gap_df["norm_interest"] = (gap_df["avg_interest"] - gap_df["avg_interest"].min()) / \
                                   (gap_df["avg_interest"].max() - gap_df["avg_interest"].min())
        gap_df["norm_listings"] = (gap_df["avg_listings"] - gap_df["avg_listings"].min()) / \
                                   (gap_df["avg_listings"].max() - gap_df["avg_listings"].min())
        gap_df["gap_score"] = gap_df["norm_interest"] - gap_df["norm_listings"]

        fig4 = px.scatter(
            gap_df,
            x="avg_interest",
            y="avg_listings",
            text="skill",
            color="gap_score",
            color_continuous_scale=[[0,"#f78166"],[0.5,"#8b949e"],[1,"#58a6ff"]],
            labels={
                "avg_interest":"Avg Google Search Interest",
                "avg_listings":"Avg Weekly Job Listings",
                "gap_score":"Gap Score (interest − listings)",
            },
        )
        fig4.update_traces(
            textposition="top center",
            textfont=dict(size=11, color="#e6edf3"),
            marker=dict(size=14),
        )
        # Add quadrant lines
        mid_x = gap_df["avg_interest"].median()
        mid_y = gap_df["avg_listings"].median()
        fig4.add_hline(y=mid_y, line_dash="dash", line_color="#30363d")
        fig4.add_vline(x=mid_x, line_dash="dash", line_color="#30363d")

        # Quadrant labels
        x_max = gap_df["avg_interest"].max()
        y_max = gap_df["avg_listings"].max()
        for text, x, y, color in [
            ("🔥 Emerging\n(high interest, low listings)",  mid_x + (x_max-mid_x)*0.05, y_max * 0.05, "#58a6ff"),
            ("🏆 Dominant\n(high interest, high listings)", mid_x + (x_max-mid_x)*0.05, y_max * 0.85, "#3fb950"),
            ("💤 Declining\n(low interest, low listings)",  0,                           y_max * 0.05, "#8b949e"),
            ("💼 Opportunity\n(low interest, high listings)",0,                          y_max * 0.85, "#ffa657"),
        ]:
            fig4.add_annotation(x=x, y=y, text=text, showarrow=False,
                                font=dict(size=10, color=color), align="left")

        fig4.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Skill Gap: Google Interest vs Job Listing Volume",
                       font=dict(color="#e6edf3", size=14)),
            height=480,
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── 5. Job title trends ────────────────────────────────────────────────────
    st.markdown("### 💼 Job Title Search Trends")
    st.caption("What job titles are people searching for most on Google?")

    if title_trends is None:
        st.info("`data/trends_jobtitles_long.csv` not found — re-run `google_trends.ipynb`.")
    else:
        all_titles = sorted(title_trends["skill"].unique())
        sel_titles = st.multiselect(
            "Select job titles to compare",
            options=all_titles,
            default=all_titles[:5],
            key="title_trends_select",
        )

        if sel_titles:
            filt_titles = title_trends[title_trends["skill"].isin(sel_titles)]

            fig5 = go.Figure()
            for i, title in enumerate(sel_titles):
                d = filt_titles[filt_titles["skill"] == title].sort_values("date")
                smoothed = d["interest"].rolling(3, center=True, min_periods=1).mean()
                color = SKILL_COLORS[i % len(SKILL_COLORS)]
                label = title.replace(" jobs", "").title()
                fig5.add_trace(go.Scatter(
                    x=d["date"], y=smoothed,
                    mode="lines", name=label,
                    line=dict(color=color, width=2.5),
                    hovertemplate=f"<b>{label}</b><br>Week: %{{x|%b %d %Y}}<br>Interest: %{{y:.0f}}<extra></extra>",
                ))
            fig5.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Job Title Search Trends (Google, US 2023–2024)",
                           font=dict(color="#e6edf3", size=14)),
            )
            fig5.update_layout(
                height=400,
                yaxis=dict(title="Interest (0–100)"),
                hovermode="x unified",
            )
            st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ── 6. Summary metrics ─────────────────────────────────────────────────────
    st.markdown("### 📋 Trends Summary Table")

    summary = (
        trends.groupby("skill")["interest"]
        .agg(["mean","max","min","std"])
        .round(1)
        .reset_index()
        .rename(columns={
            "skill":"Skill",
            "mean":"Avg Interest",
            "max":"Peak Interest",
            "min":"Min Interest",
            "std":"Volatility (std)",
        })
        .sort_values("Avg Interest", ascending=False)
    )

    # Trend direction: compare last 4 weeks vs first 4 weeks
    trend_dir = []
    for skill in summary["Skill"]:
        d = trends[trends["skill"] == skill].sort_values("date")
        if len(d) >= 8:
            early = d["interest"].head(4).mean()
            late  = d["interest"].tail(4).mean()
            delta = late - early
            if delta > 3:    trend_dir.append("📈 Rising")
            elif delta < -3: trend_dir.append("📉 Declining")
            else:            trend_dir.append("➡️ Stable")
        else:
            trend_dir.append("—")
    summary["Trend Direction"] = trend_dir

    st.dataframe(summary.reset_index(drop=True), use_container_width=True, height=380)
    st.caption(
        "Data source: Google Trends via pytrends (web scraping).  "
        "Interest is relative (0–100) within the selected time period and geography (US).  "
        "Volatility (std) measures how much weekly interest fluctuates."
    )
