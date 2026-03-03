# Mobile F2P Retention & A/B Case Study

A portfolio project simulating a real-world analytics workflow for a mobile free-to-play puzzle game (Homescapes-type). The project covers the full analysis cycle: synthetic data generation, statistical testing, behavioral segmentation, LTV proxy modeling, and an interactive analytics dashboard.

---

## Project Structure

```
f2p_analytics/
├── data/
│   ├── generate_dataset.py     # Synthetic player data generation
│   ├── players.csv             # Output: 10 000 player records
│   ├── cohort_retention.csv    # Output: 6-cohort weekly retention
│   ├── ab_daily.csv            # Output: D0-D30 A/B retention curves
│   ├── ab_results.csv          # Output: A/B test statistics
│   └── ltv_model.csv           # Output: LTV predicted vs actual
├── notebooks/
│   └── analysis.py             # Full analysis pipeline
├── dashboard/
│   └── dashboard.jsx           # Interactive React dashboard
├── requirements.txt
└── README.md
```

---

## Dataset

All data is synthetic, generated deterministically with a fixed seed (42). The dataset simulates 10 000 players across four behavioral segments with realistic retention decay, session patterns, and monetization distributions.

**Player segments:**

| Segment | Share | D7 Retention | Avg 30d Spend |
|---------|-------|--------------|---------------|
| Whale   | 5%    | ~58%         | ~$180         |
| Dolphin | 20%   | ~38%         | ~$28          |
| Minnow  | 45%   | ~22%         | ~$4           |
| Ghost   | 30%   | ~6%          | ~$0.20        |

Each player record includes: segment, A/B group, D1/D7/D30 retention, sessions in week 1, levels cleared, and 30-day spend.

---

## Analysis

**Retention analysis**
- D1, D7, D30 retention rates overall and by segment
- Churn rate at each checkpoint
- 6-cohort weekly retention heatmap

**A/B testing**
- Welch's t-test on D7 retention (control vs treatment)
- Relative uplift with 95% confidence interval
- Cohen's d effect size
- Power analysis: minimum sample size per group for target power levels (70/80/90/95%)
- Guardrail metrics: sessions, levels, conversion rate, average spend

**Behavioral segmentation**
- Segment KPI table: retention, spend, revenue share, engagement depth
- Engagement Depth Index (EDI): composite score from sessions, levels cleared, and D7 retention

**LTV proxy modeling**
- OLS regression: `spend_30d ~ EDI + ret_d7 + sessions_w1`
- Segment-level predicted vs actual comparison
- R², RMSE reported; model intended as a fast early-signal proxy, not a production model

---

## Key Results

- Treatment group shows **+9.1% uplift** on D7 retention (p < 0.001, 95% CI: [+7.4%, +10.8%])
- Whale segment (5% of users) accounts for **~62% of total revenue**
- Early signal strongest predictor of LTV: levels cleared D1-3 (r = 0.74 with 30d spend)
- Churn risk signal: players with fewer than 2 sessions in the first 3 days churn at **4.2x the baseline rate**
- LTV proxy model RMSE: $4.2 blended; Whale segment predicted within ±8%

---

## Dashboard

The dashboard (`dashboard/dashboard.jsx`) is a self-contained React component with no external data dependencies — all data is generated in-browser using the same deterministic seed as the Python pipeline.

**Tabs:**
- **Retention** — D0-D30 retention curves, cohort heatmap, session depth distribution
- **A/B Test** — uplift summary, confidence intervals, power analysis curve, day-by-day bar comparison
- **Segmentation** — radar chart, D7 retention by segment, revenue share table
- **Monetization** — LTV predicted vs actual, spend-vs-sessions scatter by segment
- **Funnel** — level progression drop-off, churn signal feature importance, A/B guardrail metrics

To run locally, paste the component into any React sandbox (StackBlitz, CodeSandbox, or a Vite project with Recharts installed).

---

## Reproduction

```bash
pip install -r requirements.txt

python data/generate_dataset.py
python notebooks/analysis.py
```

CSV outputs will be written to `data/`. All results are fully reproducible.

---

## Metrics Reference

| Metric | Definition |
|--------|-----------|
| D1/D7/D30 Retention | Share of day-0 installs still active on day N |
| Churn Rate | 1 - Retention at a given checkpoint |
| Relative Uplift | (treat - ctrl) / ctrl |
| Cohen's d | Standardized effect size for the A/B test |
| EDI | Engagement Depth Index, composite 0-100 score from sessions, levels, D7 |
| LTV Proxy | Predicted 30-day spend from early engagement signals via OLS |
| Power | Probability of detecting a true effect at a given sample size |

---

## Technologies

Python, NumPy, pandas, SciPy, React, Recharts

---

## Author

Artur — [github.com/Gobberz](https://github.com/Gobberz) — 2026
