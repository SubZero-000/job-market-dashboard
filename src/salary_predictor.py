"""
salary_predictor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, joblib
import plotly.graph_objects as go

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22", plot_bgcolor="#161b22",
    font=dict(family="Syne, sans-serif", color="#8b949e", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    colorway=["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff"],
    legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1),
)

@st.cache_resource
def load_model():
    pipe = joblib.load("model/salary_model.pkl")
    with open("model/salary_model_meta.json") as f:
        meta = json.load(f)
    return pipe, meta

def build_input(title, company, job_category, remote_status,
                work_type, experience_level, state):
    return pd.DataFrame([{
        "job_category":     job_category,
        "remote_status":    remote_status,
        "work_type":        work_type,
        "experience_level": experience_level,
        "state":            state,
        "title_clean":      title,
        "company_name":     company,
    }])

def predict(model, **kwargs):
    inp  = build_input(**kwargs)
    pred = float(np.expm1(model.predict(inp)[0]))
    return float(np.clip(pred, 20_000, 500_000))


def render_salary_predictor(df: pd.DataFrame):
    st.markdown('<div class="accent-tag">ML Model · v3</div>', unsafe_allow_html=True)
    st.title("🎯 Salary Predictor")
    st.caption(
        "Predict expected annual salary from your job profile. "
        "Model uses job title and company as primary signals via TargetEncoder."
    )

    try:
        model, meta = load_model()
    except FileNotFoundError:
        st.error(
            "⚠️ Model not found. "
            "Run `python train_salary_model.py` first to generate `model/salary_model.pkl`."
        )
        st.stop()

    # ── Model card ────────────────────────────────────────────────────────────
    with st.expander("📊 Model Performance", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm",  meta["model_name"])
        c2.metric("Test MAE",   f"${meta['test_mae']:,.0f}")
        c3.metric("Test RMSE",  f"${meta['test_rmse']:,.0f}")
        c4.metric("Test R²",    f"{meta['test_r2']:.3f}")
        st.caption(
            f"5-fold CV R²: {meta['cv_r2_mean']:.3f} ± {meta['cv_r2_std']:.3f}  |  "
            f"Encoding: {meta.get('encoder','TargetEncoder (title_clean, company_name)')}  |  "
            f"v1 R²=0.33 → v2 R²=0.35 → v3 R²={meta['test_r2']:.2f}"
        )

    st.divider()

    # ── Input form ────────────────────────────────────────────────────────────
    st.markdown("### 🧩 Your Job Profile")
    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox(
            "Job Title",
            options=meta.get("title_options", ["unknown"]),
            help="Strongest salary signal — start typing to search",
        )
        company = st.selectbox(
            "Company",
            options=["Unknown"] + meta.get("company_options", []),
            help="Second strongest signal — leave as Unknown if not applicable",
        )
        job_category = st.selectbox("Job Category", meta["job_categories"])

    with col2:
        remote_status    = st.selectbox("Work Type",        meta["remote_statuses"])
        work_type        = st.selectbox("Employment Type",  meta["work_types"])
        experience_level = st.selectbox("Experience Level", meta["experience_levels"])
        state            = st.selectbox("State (US)",       meta["states"])

    # ── Shared kwargs for all predictions ─────────────────────────────────────
    base_kwargs = dict(
        title=job_title, company=company,
        job_category=job_category, remote_status=remote_status,
        work_type=work_type, experience_level=experience_level, state=state,
    )

    predicted_salary = predict(model, **base_kwargs)

    st.divider()
    st.markdown("### 💡 Prediction")

    _, pcol, _ = st.columns([1, 2, 1])
    with pcol:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1f6feb22,#3fb95022);
                    border:1px solid #1f6feb55;border-radius:16px;
                    padding:32px;text-align:center;">
            <div style="color:#8b949e;font-size:0.8rem;letter-spacing:0.1em;
                        text-transform:uppercase;font-family:Syne,sans-serif;">
                Estimated Annual Salary
            </div>
            <div style="color:#58a6ff;font-size:3rem;font-weight:800;
                        font-family:Syne,sans-serif;margin:8px 0;">
                ${predicted_salary:,.0f}
            </div>
            <div style="color:#8b949e;font-size:0.78rem;font-family:DM Mono,monospace;">
                ±${meta['test_mae']:,.0f} typical margin of error
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(" ")

    # Confidence band
    low  = max(20_000,  predicted_salary - meta["test_mae"])
    high = min(500_000, predicted_salary + meta["test_mae"])

    fig_band = go.Figure(go.Bar(
        x=[high - low], y=[""], base=[low], orientation="h",
        marker=dict(color="#1f6feb", opacity=0.55),
        text=f"  ${low:,.0f}  ←  ${predicted_salary:,.0f}  →  ${high:,.0f}",
        textposition="inside", textfont=dict(color="#e6edf3", size=13),
    ))
    fig_band.update_layout(
        **PLOTLY_LAYOUT, height=90, showlegend=False,
        title=dict(text="Confidence Band (±MAE)",
                   font=dict(color="#e6edf3", size=12)),
    )
    fig_band.update_layout(
        xaxis=dict(tickformat="$,.0f", range=[20_000, 400_000]),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig_band, use_container_width=True)

    st.divider()

    # ── Side-by-side comparison charts ────────────────────────────────────────
    left, right = st.columns(2)

    # 1. Salary by experience level
    with left:
        st.markdown("### 📈 By Experience Level")
        bench = []
        for lvl in meta["experience_levels"]:
            kw = {**base_kwargs, "experience_level": lvl}
            bench.append({"level": lvl, "salary": predict(model, **kw)})
        bench_df = pd.DataFrame(bench)
        colors   = ["#58a6ff" if l == experience_level else "#21262d"
                    for l in bench_df["level"]]
        fig_b = go.Figure(go.Bar(
            x=bench_df["level"], y=bench_df["salary"],
            marker=dict(color=colors, line=dict(color="#30363d", width=1)),
            text=bench_df["salary"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside", textfont=dict(size=10),
        ))
        fig_b.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Predicted Salary by Experience",
                       font=dict(color="#e6edf3", size=13)),
        )
        fig_b.update_layout(
            yaxis=dict(tickformat="$,.0f"), 
            height=340,
            xaxis=dict(tickangle=-20),
        )
        
        st.plotly_chart(fig_b, use_container_width=True)

    # 2. Remote vs On-site
    with right:
        st.markdown("### 🌐 By Work Type")
        rem_rows = []
        for rs in meta["remote_statuses"]:
            kw = {**base_kwargs, "remote_status": rs}
            rem_rows.append({"work_type": rs, "salary": predict(model, **kw)})
        rem_df = pd.DataFrame(rem_rows)
        fig_r = go.Figure(go.Bar(
            x=rem_df["work_type"], y=rem_df["salary"],
            marker=dict(
                color=["#58a6ff" if r == remote_status else "#21262d"
                       for r in rem_df["work_type"]],
                line=dict(color="#30363d", width=1),
            ),
            text=rem_df["salary"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside", textfont=dict(size=12),
        ))
        fig_r.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Predicted Salary by Work Type",
                       font=dict(color="#e6edf3", size=13)),
        )
        fig_r.update_layout(
            yaxis=dict(tickformat="$,.0f"), 
            height=340,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.divider()

    # ── State comparison (top 10 states) ──────────────────────────────────────
    st.markdown("### 🗺️ Salary by State — Top 10")
    st.caption("Shows how predicted salary changes across the 10 highest-paying states for this role.")

    top_states = ["CA", "NY", "WA", "MA", "CO", "TX", "IL", "NJ", "FL", "GA"]
    state_rows = []
    for s in top_states:
        if s in meta["states"]:
            kw = {**base_kwargs, "state": s}
            state_rows.append({"state": s, "salary": predict(model, **kw)})

    state_df = pd.DataFrame(state_rows).sort_values("salary", ascending=True)
    fig_s = go.Figure(go.Bar(
        x=state_df["salary"], y=state_df["state"],
        orientation="h",
        marker=dict(
            color=["#58a6ff" if s == state else "#21262d" for s in state_df["state"]],
            line=dict(color="#30363d", width=1),
        ),
        text=state_df["salary"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside", textfont=dict(size=11),
    ))
    fig_s.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Predicted Salary by State — {job_title}",
                   font=dict(color="#e6edf3", size=13)),
    )
    fig_s.update_layout(
        xaxis=dict(tickformat="$,.0f"), 
        height=360,
    )
    st.plotly_chart(fig_s, use_container_width=True)

    st.caption(
        "Predictions are estimates based on ~28k job listings. "
        "Actual salaries vary by negotiation, team, and seniority within level."
    )


if __name__ == "__main__":
    render_salary_predictor(None)
