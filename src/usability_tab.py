"""
usability_tab.py — Usability Evaluation tab
"""

import streamlit as st
import pandas as pd
import csv
import os
import datetime

FEEDBACK_FILE = "data/feedback.csv"
HEADERS = [
    "timestamp", "participant_id", "role",
    "task1_completed", "task1_difficulty",
    "task2_completed", "task2_difficulty",
    "task3_completed", "task3_difficulty",
    "task4_completed", "task4_difficulty",
    "overall_rating", "ease_of_use", "visual_clarity",
    "would_recommend", "most_useful", "least_useful",
    "comments"
]

def save_response(row: dict):
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def load_responses():
    if not os.path.isfile(FEEDBACK_FILE):
        return pd.DataFrame()
    return pd.read_csv(FEEDBACK_FILE)


def render_usability_tab():
    st.markdown('<div class="accent-tag">Usability Evaluation</div>', unsafe_allow_html=True)
    st.title("📝 Usability Feedback")
    st.caption(
        "This form collects structured usability feedback for research purposes. "
        "Please complete the four tasks below before filling in this form."
    )

    # ── Task instructions ──────────────────────────────────────────────────────
    with st.expander("📋 Tasks to Complete First — read before filling the form", expanded=True):
        st.markdown("""
**Please complete these 4 tasks using the dashboard before filling in the form below.**

---

**Task 1 — Explorer Tab**
> Using the Explorer tab, find the top 3 companies offering the most job listings.
> Then filter by **Remote** work type only and note the average salary shown.

---

**Task 2 — Salary Predictor Tab**
> Using the Salary Predictor, predict the salary for a:
> **Senior Data Analyst** at any company, **Remote**, in **California (CA)**, with **Mid-Senior level** experience.
> Note the predicted salary and confidence range.

---

**Task 3 — Skills Trends Tab**
> Using the Skills Trends tab, identify:
> (a) The top 3 most in-demand skills overall
> (b) Which skill has the highest salary premium vs listings without it

---

**Task 4 — Market Trends Tab**
> Using the Market Trends tab:
> (a) Which skill has the highest average Google search interest?
> (b) Find one skill in the "Emerging" quadrant of the Skill Gap chart

---
""")

    st.divider()
    st.markdown("### 📊 Feedback Form")
    st.caption("All responses are anonymous. Participant ID is just a number so we can track completion.")

    with st.form("usability_form", clear_on_submit=True):

        # ── Participant info ───────────────────────────────────────────────────
        st.markdown("**About You**")
        c1, c2 = st.columns(2)
        participant_id = c1.text_input(
            "Participant ID (e.g. P1, P2)",
            placeholder="P1"
        )
        role = c2.selectbox(
            "Your background",
            ["Student — Data Science / IT",
             "Student — Other field",
             "Working professional",
             "Academic / Researcher",
             "Other"]
        )

        st.divider()

        # ── Task completion & difficulty ───────────────────────────────────────
        st.markdown("**Task Completion & Difficulty**")
        st.caption("Did you complete each task? How difficult was it? (1 = Very Easy, 5 = Very Hard)")

        tasks = [
            ("Task 1", "Find top companies + filter by Remote (Explorer tab)"),
            ("Task 2", "Predict salary for Senior Data Analyst in CA (Salary Predictor tab)"),
            ("Task 3", "Find top skills and salary premium (Skills Trends tab)"),
            ("Task 4", "Google search interest + skill gap chart (Market Trends tab)"),
        ]

        task_results = {}
        for key, label in tasks:
            st.markdown(f"*{key}: {label}*")
            tc1, tc2 = st.columns([1, 2])
            completed = tc1.radio(
                f"Completed?",
                ["Yes", "No", "Partially"],
                horizontal=True,
                key=f"{key}_completed"
            )
            difficulty = tc2.slider(
                "Difficulty (1=Easy, 5=Hard)",
                1, 5, 3,
                key=f"{key}_difficulty"
            )
            task_results[key] = (completed, difficulty)
            st.markdown("---")

        # ── Overall ratings ────────────────────────────────────────────────────
        st.markdown("**Overall Ratings**")
        r1, r2, r3 = st.columns(3)
        overall_rating = r1.slider("Overall satisfaction (1–5)", 1, 5, 3, key="overall")
        ease_of_use    = r2.slider("Ease of use (1–5)", 1, 5, 3, key="ease")
        visual_clarity = r3.slider("Visual clarity (1–5)", 1, 5, 3, key="visual")

        would_recommend = st.radio(
            "Would you recommend this dashboard to others?",
            ["Definitely yes", "Probably yes", "Not sure", "Probably not", "Definitely not"],
            horizontal=True
        )

        st.divider()

        # ── Open questions ─────────────────────────────────────────────────────
        st.markdown("**Your Thoughts**")
        most_useful  = st.text_area(
            "What did you find MOST useful about the dashboard?",
            placeholder="e.g. The salary predictor was intuitive and the confidence range was helpful...",
            height=80
        )
        least_useful = st.text_area(
            "What did you find LEAST useful or confusing?",
            placeholder="e.g. The Market Trends tab was hard to understand without context...",
            height=80
        )
        comments = st.text_area(
            "Any other comments or suggestions?",
            placeholder="e.g. Would be great to have a comparison by industry...",
            height=80
        )

        # ── Submit ─────────────────────────────────────────────────────────────
        submitted = st.form_submit_button("✅ Submit Feedback", use_container_width=True)

        if submitted:
            if not participant_id.strip():
                st.error("Please enter a Participant ID before submitting.")
            else:
                row = {
                    "timestamp":        datetime.datetime.now().isoformat(),
                    "participant_id":   participant_id.strip(),
                    "role":             role,
                    "task1_completed":  task_results["Task 1"][0],
                    "task1_difficulty": task_results["Task 1"][1],
                    "task2_completed":  task_results["Task 2"][0],
                    "task2_difficulty": task_results["Task 2"][1],
                    "task3_completed":  task_results["Task 3"][0],
                    "task3_difficulty": task_results["Task 3"][1],
                    "task4_completed":  task_results["Task 4"][0],
                    "task4_difficulty": task_results["Task 4"][1],
                    "overall_rating":   overall_rating,
                    "ease_of_use":      ease_of_use,
                    "visual_clarity":   visual_clarity,
                    "would_recommend":  would_recommend,
                    "most_useful":      most_useful,
                    "least_useful":     least_useful,
                    "comments":         comments,
                }
                save_response(row)
                st.success(f"Thank you {participant_id}! Your feedback has been saved. 🎉")
                st.balloons()

    st.divider()

    # ── Results panel (researcher view) ───────────────────────────────────────
    with st.expander("🔬 Researcher View — Summary Results", expanded=False):
        df = load_responses()

        if df.empty:
            st.info("No responses collected yet.")
        else:
            st.markdown(f"**Total responses: {len(df)}**")
            st.dataframe(df, use_container_width=True, height=300)

            st.divider()
            st.markdown("**Aggregate Metrics**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Overall Rating",   f"{df['overall_rating'].mean():.2f} / 5")
            m2.metric("Avg Ease of Use",      f"{df['ease_of_use'].mean():.2f} / 5")
            m3.metric("Avg Visual Clarity",   f"{df['visual_clarity'].mean():.2f} / 5")
            m4.metric("Would Recommend",
                      f"{(df['would_recommend'].isin(['Definitely yes','Probably yes']).sum())} / {len(df)}")

            st.markdown("**Task Completion Rates**")
            for i, (key, label) in enumerate([
                ("task1", "Task 1 — Explorer"),
                ("task2", "Task 2 — Salary Predictor"),
                ("task3", "Task 3 — Skills Trends"),
                ("task4", "Task 4 — Market Trends"),
            ], 1):
                col_name = f"{key}_completed"
                if col_name in df.columns:
                    yes_count = (df[col_name] == "Yes").sum()
                    rate = yes_count / len(df) * 100
                    avg_diff = df[f"{key}_difficulty"].mean()
                    st.markdown(
                        f"- **{label}**: {yes_count}/{len(df)} completed ({rate:.0f}%) "
                        f"| Avg difficulty: {avg_diff:.1f}/5"
                    )

            st.divider()
            st.markdown("**Download Responses**")
            st.download_button(
                "⬇️ Download feedback.csv",
                data=df.to_csv(index=False),
                file_name="feedback.csv",
                mime="text/csv",
            )
