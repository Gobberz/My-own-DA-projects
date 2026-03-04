import streamlit as st
import pandas as pd
import plotly.express as px

from lib.economy import compute_daily_econ_panel
from lib.alerts import build_alerts

st.set_page_config(page_title="Live Ops", layout="wide")
st.title("Live Ops — D1 Мониторинг и алерты экономики")

tables = st.session_state.get("TABLES", {})
mt = tables.get("market_trades")
tl = tables.get("token_ledger")

if mt is None or tl is None:
    st.error("Нужны market_trades и token_ledger (CSV). Загрузись через app.py")
    st.stop()

with st.sidebar:
    st.header("Alert settings")
    z_high = st.slider("z-threshold HIGH", 2.0, 6.0, 3.0, 0.5)
    z_low  = st.slider("z-threshold LOW", -6.0, -2.0, -3.0, 0.5)

with st.spinner("Compute daily econ panel…"):
    panel = compute_daily_econ_panel(mt, tl)

with st.spinner("Build alerts…"):
    alerts = build_alerts(panel, z_thr_high=z_high, z_thr_low=z_low)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Days", len(panel))
c2.metric("Avg DA traders", f"{panel['traders'].mean():.1f}" if "traders" in panel else "n/a")
c3.metric("Avg velocity", f"{panel['velocity'].mean():.4f}" if "velocity" in panel else "n/a")
c4.metric("Alerts", len(alerts))

st.subheader("Key charts")
cols = st.columns(2)

# Volume + traders
fig = px.line(panel, x="day", y=["volume","traders"], title="Volume & Traders (daily)")
cols[0].plotly_chart(fig, use_container_width=True)

# Velocity + net issuance
ys = [c for c in ["velocity","net_issuance"] if c in panel.columns]
if ys:
    fig2 = px.line(panel, x="day", y=ys, title="Velocity & Net Issuance (daily)")
    cols[1].plotly_chart(fig2, use_container_width=True)

cols2 = st.columns(2)
ys2 = [c for c in ["spread_p90","slippage_p90"] if c in panel.columns]
if ys2:
    fig3 = px.line(panel, x="day", y=ys2, title="Market Quality: Spread/Slippage p90")
    cols2[0].plotly_chart(fig3, use_container_width=True)

if "top1_volume_share" in panel.columns:
    fig4 = px.line(panel, x="day", y="top1_volume_share", title="Concentration: Top-1% volume share")
    cols2[1].plotly_chart(fig4, use_container_width=True)

st.subheader("Alerts")
if alerts.empty:
    st.success("Алертов нет по текущим порогам.")
else:
    st.dataframe(alerts, use_container_width=True, height=260)

    st.markdown("### Drilldown")
    # pick an alert
    pick_day = st.selectbox("День", options=alerts["day"].astype(str).unique())
    d = pd.to_datetime(pick_day)

    row = panel.loc[panel["day"] == d]
    if row.empty:
        st.info("Нет строки в panel для выбранного дня.")
    else:
        st.write("Panel snapshot:")
        st.dataframe(row, use_container_width=True)

    st.markdown("#### Compare window (±7 days)")
    w = panel[(panel["day"] >= d - pd.Timedelta(days=7)) & (panel["day"] <= d + pd.Timedelta(days=7))].copy()
    figw = px.line(w, x="day", y=[c for c in ["volume","velocity","net_issuance","spread_p90","slippage_p90","top1_volume_share"] if c in w.columns],
                   title="Window around alert day")
    st.plotly_chart(figw, use_container_width=True)
