# Job Listings Explorer & Salary Predictor

An interactive data science project that analyses US job market trends using LinkedIn job postings data. Built with Python, Streamlit, and Plotly.

---

## Project Overview

This project collects, cleans, and analyses publicly available job listing data to:

- Extract and explore in-demand skills across job roles
- Predict salary ranges using a machine learning model
- Visualise job market trends using Google Trends data
- Provide an interactive dashboard for data exploration

---

## Project Structure

```
project/
│
├── data/
│   ├── postings.csv                  # Raw LinkedIn job postings (source dataset)
│   ├── cleaned_jobs2.csv             # Cleaned and enriched dataset (output of finals.ipynb)
│   ├── trends_combined.csv           # Google Trends skill interest data (output of google_trends.ipynb)
│   ├── trends_jobtitles_long.csv     # Google Trends job title interest data
│   └── skill_weekly_listings.csv     # Weekly listing counts per skill
│
├── model/
│   ├── salary_model.pkl              # Trained salary prediction model (output of train_salary_model.py)
│   └── salary_model_meta.json        # Model metadata and dropdown options
│
├── finals.ipynb                      # Dataset 1: Data cleaning, NLP, feature engineering
├── google_trends.ipynb               # Dataset 2: Google Trends web scraping via pytrends
├── train_salary_model.py             # Model training script
│
├── ap.py                             # Main Streamlit app
├── salary_predictor.py               # Salary Predictor tab
├── skills_tab.py                     # Skills Trends tab
├── trends_tab.py                     # Market Trends tab (Google Trends)
│
└── README.md                         # This file
```

---

## Datasets

### Dataset 1 — LinkedIn Job Postings
- **Source:** Publicly available LinkedIn job postings dataset (Kaggle)
- **Raw file:** `data/postings.csv` — 123,849 job listings
- **Cleaned file:** `data/cleaned_jobs2.csv` — 35,563 listings after filtering
- **Processing:** `finals.ipynb`

Key fields: `company_name`, `title_clean`, `job_category`, `normalized_salary`, `remote_status`, `location`, `extracted_skills`, `listed_month`

### Dataset 2 — Google Trends (via pytrends)
- **Source:** Google Trends — scraped programmatically using the `pytrends` library
- **Method:** Web scraping via unofficial Google Trends API wrapper
- **Coverage:** Weekly search interest (0–100) per skill/job title, US, 2023–2024
- **Processing:** `google_trends.ipynb`

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn joblib nltk pytrends
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

### 3. Add the raw data
Place `postings.csv` into the `data/` folder.

---

## How to Run

### Step 1 — Clean the data and extract features
Run `finals.ipynb` end to end. This produces `data/cleaned_jobs2.csv` with:
- Normalised job titles (15-step regex pipeline)
- 15 job categories (expanded from 7)
- Extracted skills (150+ skills across 6 categories)
- Time features (`listed_date`, `listed_month`, `listed_week`)

### Step 2 — Train the salary model
```bash
python train_salary_model.py
```

Expected output:
```
MAE:  ~$23,500
RMSE: ~$38,700
R²:   ~0.53
CV R²: ~0.63 ± 0.01
```

Saves `model/salary_model.pkl` and `model/salary_model_meta.json`.

### Step 3 — Fetch Google Trends data
Run `google_trends.ipynb` on your **local machine** (not a cloud/server environment — Google blocks datacenter IPs).

> If you get a 429 rate limit error, wait 60 seconds and retry the cell. Increase `sleep_secs` to 15+ if errors persist.

Saves trend CSVs to `data/`.

### Step 4 — Launch the dashboard
```bash
streamlit run ap.py
```

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| 📋 Explorer | Filter and explore job listings by title, company, salary, location, and work type. 6 interactive Plotly charts. |
| 🎯 Salary Predictor | Predict annual salary from job title, company, experience level, state, and work type. Shows confidence band, salary by experience, and state comparison. |
| 🧠 Skills Trends | NLP-extracted skill demand analysis. Heatmap, category breakdown, skill deep-dive, and time trends. |
| 📈 Market Trends | Google Trends search interest vs actual job listing volume. Skill gap quadrant chart, rising/declining skills, correlation analysis. |

---

## Machine Learning Model

**Algorithm:** Gradient Boosting Regressor (scikit-learn)

**Features:**

| Feature | Type | Encoding |
|---|---|---|
| `title_clean` | Categorical (high cardinality) | TargetEncoder |
| `company_name` | Categorical (high cardinality) | TargetEncoder |
| `job_category` | Categorical | OrdinalEncoder |
| `remote_status` | Categorical | OrdinalEncoder |
| `work_type` | Categorical | OrdinalEncoder |
| `experience_level` | Categorical | OrdinalEncoder |
| `state` | Categorical | OrdinalEncoder |

**TargetEncoder** encodes each category as its mean log-salary in the training data. This is critical for high-cardinality columns like `title_clean` (19k+ unique values) and `company_name` (10k+ unique values) where ordinal encoding would be meaningless.

**Model evolution:**

| Version | Key change | Test R² | Test MAE |
|---|---|---|---|
| v1 | Baseline (5 features, OrdinalEncoder) | 0.33 | $30,700 |
| v2 | Added `title_clean`, skill flags | 0.35 | $28,800 |
| v3 | Added `company_name`, TargetEncoder | 0.53 | $23,400 |

**Cross-validation:** 5-fold CV R² = 0.63 ± 0.005 — stable across all folds.

---

## NLP Skill Extraction

Skills are extracted from job description tokens using keyword matching against a dictionary of 150+ skills across 6 categories:

| Category | Examples |
|---|---|
| Programming | python, java, javascript, scala, rust |
| Data & Analytics | sql, tableau, power bi, snowflake, dbt |
| Cloud & DevOps | aws, azure, docker, kubernetes, terraform |
| AI / ML | machine learning, tensorflow, pytorch, llm |
| Tools & Platforms | jira, salesforce, figma, microsoft office |
| Soft Skills | communication, leadership, project management |

Multi-word skills (e.g. `machine learning`, `power bi`) are matched on the full joined token string, not individual words.

---

## Title Normalisation

Raw job titles contain significant noise. A 15-step regex pipeline reduces 72,521 unique raw titles to ~57,000 normalised titles (~21% reduction):

- Removes parenthetical content: `Staff Accountant (26391)` → `staff accountant`
- Removes salary noise: `Account Executive - $19.2/hr` → `account executive`
- Removes location suffixes: `Account Executive - Buffalo` → `account executive`
- Removes work-type suffixes: `Azure Data Engineer (Full Time)` → `azure data engineer`
- Expands abbreviations: `Sr. Software Engineer` → `senior software engineer`
- Deduplicates: `RN - Registered Nurse` → `registered nurse`

---

## Job Categories

15 categories (expanded from original 7). "Other" reduced from 61.7% to 27.6%:

Software / IT · Data & Analytics · Sales · Marketing & Creative · Healthcare · Finance & Accounting · HR · Operations & Admin · Legal · Engineering · Logistics & Warehouse · Retail & Hospitality · Skilled Trades · Management · Other

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.4.0
joblib>=1.3.0
nltk>=3.8.0
pytrends>=4.9.2
urllib3<2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Known Limitations

- **Salary model R² ~0.53** — remaining variance is explained by factors not in the dataset: company-specific pay bands, exact years of experience within a level, individual negotiation, and specific skill depth.
- **Google Trends rate limiting** — pytrends is subject to 429 errors from Google. Increase sleep delays or export data manually from trends.google.com if automated fetching fails.
- **US-centric dataset** — the LinkedIn postings dataset is predominantly US-based. Insights may not generalise to other markets.
- **Skill extraction is keyword-based** — it detects presence/absence of skill terms but cannot assess proficiency level or contextual relevance.

---

## Author

Student ID: 25011550  
Massey University
