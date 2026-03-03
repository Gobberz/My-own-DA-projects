"""
generate_dataset.py
Synthetic player dataset for Mobile F2P retention & A/B case study.
Outputs: players.csv, cohort_retention.csv, ab_daily.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_PLAYERS = 50_000
OUTPUT_DIR = Path(__file__).parent

rng = np.random.default_rng(SEED)


SEGMENTS = ["Whale", "Dolphin", "Minnow", "Ghost"]
SEG_WEIGHTS = [0.05, 0.20, 0.45, 0.30]

SEG_PARAMS = {
    "Whale":   dict(d1=0.82, d7=0.58, d30=0.32, spend_mu=180, spend_sd=60,  sessions_mu=12, levels_mu=45),
    "Dolphin": dict(d1=0.68, d7=0.38, d30=0.16, spend_mu=28,  spend_sd=12,  sessions_mu=6,  levels_mu=22),
    "Minnow":  dict(d1=0.52, d7=0.22, d30=0.07, spend_mu=4,   spend_sd=2,   sessions_mu=3,  levels_mu=8),
    "Ghost":   dict(d1=0.25, d7=0.06, d30=0.01, spend_mu=0.2, spend_sd=0.1, sessions_mu=1,  levels_mu=2),
}

TREATMENT_LIFT = 1.09  # +9% on retention for treatment group


def _clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_players():
    segments = rng.choice(SEGMENTS, size=N_PLAYERS, p=SEG_WEIGHTS)
    groups   = rng.choice(["control", "treatment"], size=N_PLAYERS)

    rows = []
    for seg, group in zip(segments, groups):
        p = SEG_PARAMS[seg]
        lift = TREATMENT_LIFT if group == "treatment" else 1.0

        d1  = float(_clamp(rng.normal(p["d1"] * lift, 0.06), 0, 1))
        d7  = float(_clamp(rng.normal(p["d7"] * lift, 0.05), 0, d1))
        d30 = float(_clamp(rng.normal(p["d30"] * lift, 0.03), 0, d7))

        if seg == "Ghost":
            spend = 0.0
        elif seg == "Minnow":
            spend = float(_clamp(rng.normal(p["spend_mu"], p["spend_sd"]), 0, 15)) if rng.random() < 0.10 else 0.0
        else:
            spend = float(_clamp(rng.normal(p["spend_mu"], p["spend_sd"]), 0, 700))

        sessions = int(_clamp(rng.normal(p["sessions_mu"], 2), 0, 40))
        levels   = int(_clamp(rng.normal(p["levels_mu"],  8), 0, 120))

        rows.append(dict(
            segment=seg,
            ab_group=group,
            ret_d1=round(d1, 4),
            ret_d7=round(d7, 4),
            ret_d30=round(d30, 4),
            sessions_w1=sessions,
            levels_cleared=levels,
            spend_30d=round(spend, 2),
        ))

    return pd.DataFrame(rows)


def generate_cohort_retention():
    cohorts = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    weeks   = [1, 2, 3, 4]
    rows = []
    for ci, cohort in enumerate(cohorts):
        for wi, week in enumerate(weeks):
            base = 65 - wi * 14 + ci * 1.5
            retention = float(_clamp(rng.normal(base, 3), 3, 97))
            rows.append(dict(cohort=cohort, week=week, retention_pct=round(retention, 1)))
    return pd.DataFrame(rows)


def generate_ab_daily():
    days = list(range(0, 31))
    rows = []
    for d in days:
        for group in ["control", "treatment"]:
            lift = TREATMENT_LIFT if group == "treatment" else 1.0
            base = np.exp(-0.18 * d) * (0.85 + 0.05 * np.exp(-0.5 * d)) if d > 0 else 1.0
            ret  = round(base * lift * 100, 2)
            rows.append(dict(day=d, ab_group=group, retention_pct=ret))
    return pd.DataFrame(rows)


def main():
    players = generate_players()
    cohorts = generate_cohort_retention()
    ab_daily = generate_ab_daily()

    players.to_csv(OUTPUT_DIR / "players.csv", index=False)
    cohorts.to_csv(OUTPUT_DIR / "cohort_retention.csv", index=False)
    ab_daily.to_csv(OUTPUT_DIR / "ab_daily.csv", index=False)

    print(f"players.csv        {len(players):>6} rows")
    print(f"cohort_retention   {len(cohorts):>6} rows")
    print(f"ab_daily           {len(ab_daily):>6} rows")

    seg_summary = players.groupby("segment").agg(
        count=("segment", "size"),
        d7_ret=("ret_d7", "mean"),
        avg_spend=("spend_30d", "mean"),
    ).round(3)
    print("\nSegment summary:")
    print(seg_summary.to_string())


if __name__ == "__main__":
    main()
