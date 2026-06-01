"""
skills_tab.py
"""

import ast
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Syne, sans-serif", color="#8b949e", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", tickfont=dict(color="#8b949e")),
    colorway=["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff"],
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1),
)

SKILL_DICT = {
    "Programming":      ["python","java","javascript","typescript","c++","c#","go",
                         "ruby","php","swift","kotlin","scala","rust",
                         "perl","matlab","bash","shell","vba","groovy"],
    "Data & Analytics": ["sql","excel","tableau","power bi","pandas","numpy",
                         "spark","hadoop","dbt","looker","sas","spss","alteryx",
                         "databricks","snowflake","redshift","bigquery","airflow",
                         "etl","data warehouse","data pipeline"],
    "Cloud & DevOps":   ["aws","azure","gcp","docker","kubernetes","terraform",
                         "jenkins","linux","ansible","git","github","gitlab",
                         "ci/cd","devops","microservices","rest api","graphql",
                         "nginx","redis","kafka"],
    "AI / ML":          ["machine learning","deep learning","nlp","tensorflow",
                         "pytorch","scikit-learn","computer vision","llm",
                         "neural network","reinforcement learning","hugging face",
                         "langchain","openai","generative ai","prompt engineering",
                         "feature engineering","model deployment"],
    "Tools & Platforms":["jira","confluence","salesforce","sap","quickbooks",
                         "servicenow","hubspot","figma","adobe","autocad",
                         "microsoft office","sharepoint","powerpoint","outlook",
                         "slack","zoom","trello","asana","monday"],
    "Soft Skills":      ["communication","leadership","teamwork","collaboration",
                         "problem solving","critical thinking","project management",
                         "presentation","negotiation","mentoring","time management",
                         "attention to detail","customer service","stakeholder management"],
}

ALL_SKILLS = {skill: cat for cat, skills in SKILL_DICT.items() for skill in skills}
SKILL_LIST = list(ALL_SKILLS.keys())

CAT_COLORS = {
    "Programming":      "#58a6ff",
    "Data & Analytics": "#3fb950",
    "Cloud & DevOps":   "#ffa657",
    "AI / ML":          "#d2a8ff",
    "Tools & Platforms":"#f78166",
    "Soft Skills":      "#79c0ff",
}

# ── Parse helpers ──────────────────────────────────────────────────────────────
def parse_list_col(val):
    if isinstance(val, list): return val
    if pd.isna(val): return []
    try: return ast.literal_eval(str(val))
    except: return []

@st.cache_data(show_spinner="Preparing skills data…")
def prepare_skills(df_hash, tokens_series, skills_series, has_presaved):
    if has_presaved:
        return skills_series.apply(parse_list_col)
    # Fallback: re-extract from tokens
    def match(tokens):
        text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        return [s for s in SKILL_LIST if s in text]
    return tokens_series.apply(
        lambda v: match(parse_list_col(v))
    )


def render_skills_tab(df: pd.DataFrame):
    st.markdown('<div class="accent-tag">NLP Analysis · v2</div>', unsafe_allow_html=True)
    st.title("🧠 Skills Trend Analysis")
    st.caption("Skills extracted from job descriptions. Filter to explore demand, category breakdown, and time trends.")

    df = df.copy()

    # Use pre-saved skills if available, else fallback to token extraction
    has_presaved = "extracted_skills" in df.columns and df["extracted_skills"].notna().any()
    df["_skills"] = prepare_skills(
        str(len(df)),
        df.get("tokens", pd.Series(dtype=str)),
        df.get("extracted_skills", pd.Series(dtype=str)),
        has_presaved,
    )

    df_has = df[df["_skills"].map(len) > 0].copy()
    pct    = len(df_has) / len(df) * 100
    st.caption(f"Skills found in {len(df_has):,} of {len(df):,} listings ({pct:.1f}%) — "
               f"{'using pre-saved skills column ✅' if has_presaved else 'extracted from tokens at runtime'}")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("### 🔧 Filter")
    fc1, fc2, fc3 = st.columns(3)

    sel_categories = fc1.multiselect(
        "Job Category", sorted(df["job_category"].dropna().unique()), placeholder="All categories")
    sel_remote     = fc2.multiselect(
        "Work Type", sorted(df["remote_status"].dropna().unique()),
        default=sorted(df["remote_status"].dropna().unique()))
    top_n = fc3.slider("Top N skills", 5, 30, 15)

    filt = df_has.copy()
    if sel_categories: filt = filt[filt["job_category"].isin(sel_categories)]
    if sel_remote:     filt = filt[filt["remote_status"].isin(sel_remote)]

    if filt.empty:
        st.warning("No data matches selected filters.")
        return

    st.divider()

    # ── 1. Top skills bar ─────────────────────────────────────────────────────
    st.markdown("### 📊 Most In-Demand Skills")

    all_ex      = filt["_skills"].explode().dropna()
    skill_counts = all_ex.value_counts().head(top_n).reset_index()
    skill_counts.columns = ["skill", "count"]
    skill_counts["category"] = skill_counts["skill"].map(ALL_SKILLS)
    skill_counts["color"]    = skill_counts["category"].map(CAT_COLORS).fillna("#8b949e")
    skill_counts["pct"]      = (skill_counts["count"] / len(filt) * 100).round(1)

    fig1 = go.Figure(go.Bar(
        x=skill_counts["count"], y=skill_counts["skill"],
        orientation="h",
        marker=dict(color=skill_counts["color"], line=dict(width=0)),
        text=skill_counts.apply(lambda r: f"{r['count']:,}  ({r['pct']}%)", axis=1),
        textposition="outside", textfont=dict(color="#8b949e", size=11),
        customdata=skill_counts["category"],
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>Category: %{customdata}<extra></extra>",
    ))
    fig1.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Top {top_n} Skills by Mention Count", font=dict(color="#e6edf3", size=14)),
    )
    fig1.update_layout(
        height=max(360, top_n * 28), 
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Category legend
    leg = st.columns(len(CAT_COLORS))
    for i, (cat, color) in enumerate(CAT_COLORS.items()):
        leg[i].markdown(f'<span style="color:{color};font-size:0.75rem;">■ {cat}</span>',
                        unsafe_allow_html=True)

    st.divider()

    # ── 2. Category breakdown ─────────────────────────────────────────────────
    st.markdown("### 🗂️ Skill Demand by Category")

    cat_counts = (
        all_ex.map(ALL_SKILLS).value_counts().reset_index()
    )
    cat_counts.columns = ["category", "count"]
    cat_counts["color"] = cat_counts["category"].map(CAT_COLORS)

    fig2 = go.Figure(go.Bar(
        x=cat_counts["category"], y=cat_counts["count"],
        marker=dict(color=cat_counts["color"], line=dict(width=0)),
        text=cat_counts["count"], textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Total Skill Mentions by Category", font=dict(color="#e6edf3", size=14)),
        height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 3. Heatmap — skills by job category ───────────────────────────────────
    st.markdown("### 🔥 Skills Heatmap by Job Category")

    top_skill_list = skill_counts["skill"].tolist()
    exploded = filt[["job_category","_skills"]].explode("_skills").dropna()
    exploded = exploded[exploded["_skills"].isin(top_skill_list)]

    pivot = (
        exploded.groupby(["job_category","_skills"]).size().reset_index(name="count")
        .pivot(index="job_category", columns="_skills", values="count").fillna(0)
    )
    job_totals = filt.groupby("job_category").size()
    pivot_pct  = (pivot.div(job_totals, axis=0) * 100).fillna(0)
    pivot_pct  = pivot_pct.loc[:, pivot_pct.max() > 1]

    fig3 = go.Figure(go.Heatmap(
        z=pivot_pct.values, x=pivot_pct.columns.tolist(), y=pivot_pct.index.tolist(),
        colorscale=[[0,"#0d1117"],[0.3,"#1f3a5f"],[0.7,"#1f6feb"],[1,"#58a6ff"]],
        hovertemplate="<b>%{y}</b><br>Skill: %{x}<br>%{z:.1f}% of listings<extra></extra>",
        text=np.round(pivot_pct.values, 1), texttemplate="%{z:.0f}%",
        textfont=dict(size=10),
    ))
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="% of Listings Mentioning Each Skill (by Job Category)",
                   font=dict(color="#e6edf3", size=14)),
        height=max(300, len(pivot_pct) * 52 + 80),
    )
    fig3.update_layout(
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── 4. Time trends ────────────────────────────────────────────────────────
    st.markdown("### 📅 Skill Demand Over Time")

    has_date = "listed_month" in filt.columns and filt["listed_month"].notna().any()
    if not has_date:
        st.info("Time trend data not available — re-run `finals.ipynb` to add `listed_month` to the CSV.")
    else:
        top5_skills = skill_counts["skill"].head(5).tolist()

        time_exploded = filt[["listed_month","_skills"]].explode("_skills").dropna()
        time_exploded = time_exploded[time_exploded["_skills"].isin(top5_skills)]

        # Count per skill per month
        time_counts = (
            time_exploded.groupby(["listed_month","_skills"]).size()
            .reset_index(name="count")
            .sort_values("listed_month")
        )

        # Normalise by total listings that month to get % not raw count
        monthly_totals = filt.groupby("listed_month").size().reset_index(name="total")
        time_counts    = time_counts.merge(monthly_totals, on="listed_month")
        time_counts["pct"] = (time_counts["count"] / time_counts["total"] * 100).round(2)

        color_map = {s: list(CAT_COLORS.values())[i] for i, s in enumerate(top5_skills)}

        fig_time = go.Figure()
        for skill in top5_skills:
            d = time_counts[time_counts["_skills"] == skill]
            fig_time.add_trace(go.Scatter(
                x=d["listed_month"], y=d["pct"],
                mode="lines+markers",
                name=skill.title(),
                line=dict(color=color_map.get(skill, "#58a6ff"), width=2),
                marker=dict(size=6),
                hovertemplate=f"<b>{skill.title()}</b><br>Month: %{{x}}<br>%{{y:.1f}}% of listings<extra></extra>",
            ))
        fig_time.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Top 5 Skills — % of Monthly Listings Mentioning Each Skill",
                       font=dict(color="#e6edf3", size=14)),
        )
        fig_time.update_layout(
            height=380,
            xaxis=dict(title="Month"),
            yaxis=dict(title="% of Listings", ticksuffix="%"),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    st.divider()

    # ── 5. Remote vs On-site ──────────────────────────────────────────────────
    st.markdown("### 🌐 Remote vs On-site Skill Demand")

    rem_ex = filt[["remote_status","_skills"]].explode("_skills").dropna()
    rem_ex = rem_ex[rem_ex["_skills"].isin(top_skill_list[:20])]
    rem_pivot = (
        rem_ex.groupby(["_skills","remote_status"]).size()
        .reset_index(name="count")
    )

    fig4 = px.bar(
        rem_pivot, x="_skills", y="count", color="remote_status",
        barmode="group",
        color_discrete_map={"Remote":"#58a6ff","On-site":"#f78166"},
        labels={"_skills":"Skill","count":"Mentions","remote_status":"Work Type"},
    )
    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Skill Mentions: Remote vs On-site", font=dict(color="#e6edf3", size=14)),
    )
    fig4.update_layout(
        height=380, 
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── 6. Skill deep-dive ────────────────────────────────────────────────────
    st.markdown("### 🔍 Skill Deep-Dive")

    dc1, dc2 = st.columns([1, 3])
    chosen = dc1.selectbox(
        "Pick a skill", sorted(SKILL_LIST),
        index=sorted(SKILL_LIST).index("python") if "python" in SKILL_LIST else 0,
    )

    skill_df = filt[filt["_skills"].apply(lambda s: chosen in s)]
    by_cat   = skill_df["job_category"].value_counts().reset_index()
    by_cat.columns = ["job_category","count"]

    with dc2:
        fig5 = go.Figure(go.Bar(
            x=by_cat["count"], y=by_cat["job_category"],
            orientation="h",
            marker=dict(color="#3fb950", line=dict(width=0)),
            text=by_cat["count"], textposition="outside",
            textfont=dict(color="#8b949e", size=11),
        ))
        fig5.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f'"{chosen}" — demand by job category',
                       font=dict(color="#e6edf3", size=13)),
        )
        fig5.update_layout(
            height=300, 
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig5, use_container_width=True)

    has_skill = filt[filt["_skills"].apply(lambda s: chosen in s)]["normalized_salary"].dropna()
    no_skill  = filt[filt["_skills"].apply(lambda s: chosen not in s)]["normalized_salary"].dropna()
    delta     = has_skill.median() - no_skill.median()

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Listings mentioning skill", f"{len(skill_df):,}")
    mc2.metric("Median salary (with skill)", f"${has_skill.median():,.0f}" if not has_skill.empty else "N/A")
    mc3.metric("Salary premium vs without",  f"${delta:+,.0f}" if not has_skill.empty else "N/A")

    st.divider()

    # ── 7. Full frequency table ───────────────────────────────────────────────
    st.markdown("### 📋 Full Skill Frequency Table")

    full = all_ex.value_counts().reset_index()
    full.columns = ["Skill","Mentions"]
    full["Category"]     = full["Skill"].map(ALL_SKILLS)
    full["% of Listings"] = (full["Mentions"] / len(filt) * 100).round(1).astype(str) + "%"

    st.dataframe(full, use_container_width=True, height=360)
    st.caption(f"Based on {len(filt):,} filtered listings.")
