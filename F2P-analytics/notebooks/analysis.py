"""
analysis.py
Retention analysis, A/B testing, segmentation, and LTV proxy modeling.
Run after generate_dataset.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    players  = pd.read_csv(DATA_DIR / "players.csv")
    cohorts  = pd.read_csv(DATA_DIR / "cohort_retention.csv")
    ab_daily = pd.read_csv(DATA_DIR / "ab_daily.csv")
    return players, cohorts, ab_daily


# ─── RETENTION ───────────────────────────────────────────────────────────────

def retention_summary(players: pd.DataFrame) -> pd.DataFrame:
    overall = players[["ret_d1", "ret_d7", "ret_d30"]].mean().rename({
        "ret_d1": "D1", "ret_d7": "D7", "ret_d30": "D30"
    })
    by_seg = players.groupby("segment")[["ret_d1", "ret_d7", "ret_d30"]].mean()
    return overall, by_seg


def churn_rate(players: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "D1_churn":  1 - players["ret_d1"].mean(),
        "D7_churn":  1 - players["ret_d7"].mean(),
        "D30_churn": 1 - players["ret_d30"].mean(),
    }).round(4)


# ─── A/B TESTING ─────────────────────────────────────────────────────────────

def ab_test_d7(players: pd.DataFrame) -> dict:
    ctrl  = players.loc[players["ab_group"] == "control",    "ret_d7"]
    treat = players.loc[players["ab_group"] == "treatment",  "ret_d7"]

    t_stat, p_val = stats.ttest_ind(treat, ctrl, equal_var=False)

    mu_ctrl  = ctrl.mean()
    mu_treat = treat.mean()
    uplift   = (mu_treat - mu_ctrl) / mu_ctrl

    se = np.sqrt(ctrl.var() / len(ctrl) + treat.var() / len(treat))
    ci_lo = (mu_treat - mu_ctrl - 1.96 * se) / mu_ctrl
    ci_hi = (mu_treat - mu_ctrl + 1.96 * se) / mu_ctrl

    effect_size = (mu_treat - mu_ctrl) / np.sqrt(
        (ctrl.std() ** 2 + treat.std() ** 2) / 2
    )

    return dict(
        n_control=len(ctrl),
        n_treatment=len(treat),
        mean_control=round(mu_ctrl, 4),
        mean_treatment=round(mu_treat, 4),
        uplift_pct=round(uplift * 100, 2),
        ci_95_lo=round(ci_lo * 100, 2),
        ci_95_hi=round(ci_hi * 100, 2),
        t_statistic=round(t_stat, 4),
        p_value=round(p_val, 6),
        cohens_d=round(effect_size, 4),
        significant=p_val < 0.05,
    )


def power_analysis(effect_size: float, alpha: float = 0.05) -> dict:
    """
    Minimum sample size per group for target powers using normal approximation.
    effect_size: Cohen's d
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    results = {}
    for power in [0.70, 0.80, 0.90, 0.95]:
        z_beta = stats.norm.ppf(power)
        n = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
        results[f"power_{int(power*100)}"] = n
    return results


def guardrail_metrics(players: pd.DataFrame) -> pd.DataFrame:
    grp = players.groupby("ab_group")
    metrics = pd.DataFrame({
        "avg_sessions":    grp["sessions_w1"].mean(),
        "avg_levels":      grp["levels_cleared"].mean(),
        "pct_spenders":    grp["spend_30d"].apply(lambda x: (x > 0).mean()),
        "avg_spend":       grp["spend_30d"].mean(),
    }).round(3)
    return metrics


# ─── SEGMENTATION ────────────────────────────────────────────────────────────

def segment_kpis(players: pd.DataFrame) -> pd.DataFrame:
    total_revenue = players["spend_30d"].sum()

    agg = players.groupby("segment").agg(
        count=("segment", "size"),
        d1_ret=("ret_d1", "mean"),
        d7_ret=("ret_d7", "mean"),
        d30_ret=("ret_d30", "mean"),
        avg_sessions=("sessions_w1", "mean"),
        avg_levels=("levels_cleared", "mean"),
        avg_spend=("spend_30d", "mean"),
        total_revenue=("spend_30d", "sum"),
    ).round(3)

    agg["share_pct"]    = (agg["count"] / len(players) * 100).round(1)
    agg["rev_share_pct"] = (agg["total_revenue"] / total_revenue * 100).round(1)
    return agg


def engagement_depth_index(players: pd.DataFrame) -> pd.Series:
    """
    Composite score: 0-100 from sessions + levels + D7 retention.
    Useful as a single early-signal feature for LTV proxy.
    """
    s = players["sessions_w1"]
    l = players["levels_cleared"]
    r = players["ret_d7"]

    score = (
        (s - s.min()) / (s.max() - s.min()) * 0.35 +
        (l - l.min()) / (l.max() - l.min()) * 0.35 +
        r * 0.30
    ) * 100
    return score.round(2)


# ─── LTV PROXY MODEL ─────────────────────────────────────────────────────────

def ltv_proxy_model(players: pd.DataFrame) -> dict:
    """
    Linear regression: spend_30d ~ edi + ret_d7 + sessions_w1
    Returns coefficients, R^2, and per-segment predicted vs actual.
    """
    from numpy.linalg import lstsq

    df = players.copy()
    df["edi"] = engagement_depth_index(df)

    X = np.column_stack([
        np.ones(len(df)),
        df["edi"],
        df["ret_d7"],
        df["sessions_w1"],
    ])
    y = df["spend_30d"].values

    coef, _, _, _ = lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(((y - y_pred) ** 2).mean())

    df["predicted_ltv"] = np.clip(y_pred, 0, None).round(2)
    seg_compare = df.groupby("segment").agg(
        actual_ltv=("spend_30d", "mean"),
        predicted_ltv=("predicted_ltv", "mean"),
    ).round(2)
    seg_compare["error_pct"] = (
        (seg_compare["predicted_ltv"] - seg_compare["actual_ltv"]).abs()
        / seg_compare["actual_ltv"].replace(0, np.nan) * 100
    ).round(1)

    return dict(
        intercept=round(coef[0], 4),
        coef_edi=round(coef[1], 4),
        coef_d7_ret=round(coef[2], 4),
        coef_sessions=round(coef[3], 4),
        r_squared=round(r2, 4),
        rmse=round(rmse, 4),
        segment_comparison=seg_compare,
    )


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    players, cohorts, ab_daily = load_data()

    print("=" * 60)
    print("RETENTION SUMMARY")
    print("=" * 60)
    overall, by_seg = retention_summary(players)
    print("Overall:", overall.to_dict())
    print("\nBy segment:\n", by_seg.to_string())
    print("\nChurn rates:", churn_rate(players).to_dict())

    print("\n" + "=" * 60)
    print("A/B TEST — D7 RETENTION")
    print("=" * 60)
    ab = ab_test_d7(players)
    for k, v in ab.items():
        print(f"  {k:20s}: {v}")

    print("\nPower analysis (effect size = Cohen's d):")
    pwr = power_analysis(abs(ab["cohens_d"]))
    for k, v in pwr.items():
        print(f"  {k}: {v} per group")

    print("\nGuardrail metrics:")
    print(guardrail_metrics(players).to_string())

    print("\n" + "=" * 60)
    print("SEGMENTATION KPIs")
    print("=" * 60)
    print(segment_kpis(players).to_string())

    print("\n" + "=" * 60)
    print("LTV PROXY MODEL")
    print("=" * 60)
    ltv = ltv_proxy_model(players)
    for k, v in ltv.items():
        if k != "segment_comparison":
            print(f"  {k}: {v}")
    print("\nSegment comparison:")
    print(ltv["segment_comparison"].to_string())

    results = {
        "ab_results": {k: v for k, v in ab.items() if k != "significant"},
        "power_analysis": pwr,
        "ltv_model": {k: v for k, v in ltv.items() if k != "segment_comparison"},
    }
    pd.DataFrame([results["ab_results"]]).to_csv(DATA_DIR / "ab_results.csv", index=False)
    ltv["segment_comparison"].to_csv(DATA_DIR / "ltv_model.csv")
    print("\nResults saved to data/")


if __name__ == "__main__":
    main()
