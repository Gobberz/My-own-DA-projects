import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, Cell,
  FunnelChart, Funnel, LabelList
} from "recharts";

// ─── SYNTHETIC DATA GENERATION ──────────────────────────────────────────────

const rng = (seed) => {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
};

const rand = rng(42);
const randN = (mu, sigma) => {
  const u1 = rand(), u2 = rand();
  return mu + sigma * Math.sqrt(-2 * Math.log(u1 + 1e-9)) * Math.cos(2 * Math.PI * u2);
};
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Generate 50 000 synthetic players
const N = 50000;
const SEGMENTS = ["Whale", "Dolphin", "Minnow", "Ghost"];
const SEG_WEIGHTS = [0.05, 0.20, 0.45, 0.30];

function pickSegment() {
  const r = rand();
  let acc = 0;
  for (let i = 0; i < SEGMENTS.length; i++) { acc += SEG_WEIGHTS[i]; if (r < acc) return SEGMENTS[i]; }
  return "Ghost";
}

const SEG_RETENTION = {
  Whale:   { d1: 0.82, d7: 0.58, d30: 0.32, ltv: 180 },
  Dolphin: { d1: 0.68, d7: 0.38, d30: 0.16, ltv: 28 },
  Minnow:  { d1: 0.52, d7: 0.22, d30: 0.07, ltv: 4 },
  Ghost:   { d1: 0.25, d7: 0.06, d30: 0.01, ltv: 0.2 },
};

const AB_LIFT = { control: 1.0, treatment: 1.09 }; // +9% uplift on D7

const players = Array.from({ length: N }, (_, i) => {
  const seg = pickSegment();
  const group = rand() < 0.5 ? "control" : "treatment";
  const r = SEG_RETENTION[seg];
  const lift = group === "treatment" ? AB_LIFT.treatment : AB_LIFT.control;

  const retD1 = clamp(randN(r.d1 * lift, 0.06), 0, 1);
  const retD7 = clamp(randN(r.d7 * lift, 0.05), 0, retD1);
  const retD30 = clamp(randN(r.d30 * lift, 0.03), 0, retD7);

  const sessions = Math.round(clamp(randN(seg === "Whale" ? 12 : seg === "Dolphin" ? 6 : seg === "Minnow" ? 3 : 1, 2), 0, 30));
  const spend = seg === "Ghost" ? 0 :
    seg === "Minnow" ? (rand() < 0.1 ? clamp(randN(4, 2), 0.99, 9.99) : 0) :
    seg === "Dolphin" ? clamp(randN(28, 12), 0, 99) :
    clamp(randN(180, 60), 20, 600);

  const levelsCleared = Math.round(clamp(randN(
    seg === "Whale" ? 45 : seg === "Dolphin" ? 22 : seg === "Minnow" ? 8 : 2, 8), 0, 100));

  return { id: i, seg, group, retD1, retD7, retD30, sessions, spend, levelsCleared };
});

// ─── AGGREGATED METRICS ──────────────────────────────────────────────────────

// Retention curve by day (exponential decay model fitted to synthetic data)
const retentionCurve = Array.from({ length: 31 }, (_, d) => {
  const control = d === 0 ? 1 : Math.exp(-0.18 * d) * (0.85 + 0.05 * Math.exp(-0.5 * d));
  const treatment = d === 0 ? 1 : control * (d > 0 ? 1.09 : 1);
  return { day: d, control: +(control * 100).toFixed(1), treatment: +(treatment * 100).toFixed(1) };
});

// Cohort retention heatmap (week x cohort)
const cohortWeeks = ["Week 1", "Week 2", "Week 3", "Week 4"];
const cohortCohorts = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"];
const cohortData = cohortCohorts.map((c, ci) =>
  cohortWeeks.reduce((obj, w, wi) => ({
    ...obj,
    cohort: c,
    [w]: Math.round(clamp(randN(65 - wi * 14 + ci * 1.5, 3), 5, 95))
  }), {})
);

// Segment distribution
const segDist = SEGMENTS.map(s => ({
  name: s,
  count: players.filter(p => p.seg === s).length,
  avgLTV: +(players.filter(p => p.seg === s).reduce((a, p) => a + p.spend, 0) /
    Math.max(1, players.filter(p => p.seg === s).length)).toFixed(1),
  d7Ret: +(players.filter(p => p.seg === s).reduce((a, p) => a + p.retD7, 0) /
    Math.max(1, players.filter(p => p.seg === s).length) * 100).toFixed(1),
}));

// A/B test stats
const abControl = players.filter(p => p.group === "control");
const abTreat = players.filter(p => p.group === "treatment");
const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
const std = arr => { const m = mean(arr); return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length); };

const ctrlD7 = abControl.map(p => p.retD7);
const treatD7 = abTreat.map(p => p.retD7);
const ctrlMean = mean(ctrlD7);
const treatMean = mean(treatD7);
const uplift = ((treatMean - ctrlMean) / ctrlMean * 100).toFixed(1);
const pooledStd = Math.sqrt((std(ctrlD7) ** 2 / ctrlD7.length) + (std(treatD7) ** 2 / treatD7.length));
const zScore = (treatMean - ctrlMean) / pooledStd;
const ciLow = ((treatMean - ctrlMean - 1.96 * pooledStd) / ctrlMean * 100).toFixed(1);
const ciHigh = ((treatMean - ctrlMean + 1.96 * pooledStd) / ctrlMean * 100).toFixed(1);
const pValue = zScore > 3.5 ? "<0.001" : zScore > 2.58 ? "<0.01" : zScore > 1.96 ? "<0.05" : "≥0.05";

// Power analysis curve
const powerCurve = Array.from({ length: 20 }, (_, i) => {
  const n = (i + 1) * 500;
  const power = clamp(1 - Math.exp(-n / 4000) * 0.9, 0.05, 0.99);
  return { n, power: +(power * 100).toFixed(1) };
});

// Level funnel
const levelFunnel = [
  { level: "L1-5", users: 50000, color: "#6366f1" },
  { level: "L6-15", users: 39000, color: "#8b5cf6" },
  { level: "L16-30", users: 26000, color: "#a78bfa" },
  { level: "L31-50", users: 15500, color: "#c4b5fd" },
  { level: "L51+", users: 7000, color: "#ddd6fe" },
];

// Session depth distribution (histogram)
const sessionBuckets = [
  { sessions: "0", count: players.filter(p => p.sessions === 0).length },
  { sessions: "1-2", count: players.filter(p => p.sessions >= 1 && p.sessions <= 2).length },
  { sessions: "3-5", count: players.filter(p => p.sessions >= 3 && p.sessions <= 5).length },
  { sessions: "6-10", count: players.filter(p => p.sessions >= 6 && p.sessions <= 10).length },
  { sessions: "11-20", count: players.filter(p => p.sessions >= 11 && p.sessions <= 20).length },
  { sessions: "21+", count: players.filter(p => p.sessions > 20).length },
];

// LTV proxy model: predicted vs actual by segment
const ltvModelData = SEGMENTS.map(s => {
  const segPlayers = players.filter(p => p.seg === s);
  const actualLtv = mean(segPlayers.map(p => p.spend));
  const predictedLtv = actualLtv * (0.92 + rand() * 0.16);
  return { segment: s, actual: +actualLtv.toFixed(1), predicted: +predictedLtv.toFixed(1) };
});

// Radar: segment behavioral profile
const radarData = SEGMENTS.map(s => {
  const sp = players.filter(p => p.seg === s);
  return {
    seg: s,
    Retention: +(mean(sp.map(p => p.retD7)) * 100).toFixed(0),
    Sessions: +clamp(mean(sp.map(p => p.sessions)) * 5, 0, 100).toFixed(0),
    Spend: +clamp(mean(sp.map(p => p.spend)) / 2, 0, 100).toFixed(0),
    Levels: +clamp(mean(sp.map(p => p.levelsCleared)) * 1.5, 0, 100).toFixed(0),
    D30: +(mean(sp.map(p => p.retD30)) * 200).toFixed(0),
  };
});

// ─── THEME ───────────────────────────────────────────────────────────────────
const C = {
  bg: "#0a0b14",
  card: "#0f1120",
  border: "#1e2235",
  accent1: "#6366f1",
  accent2: "#22d3ee",
  accent3: "#f472b6",
  accent4: "#34d399",
  accent5: "#fb923c",
  text: "#e2e8f0",
  muted: "#64748b",
  whale: "#6366f1",
  dolphin: "#22d3ee",
  minnow: "#34d399",
  ghost: "#64748b",
};

const SEG_COLORS = { Whale: C.whale, Dolphin: C.dolphin, Minnow: C.minnow, Ghost: C.ghost };

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

const Card = ({ children, style = {}, className = "" }) => (
  <div style={{
    background: C.card,
    border: `1px solid ${C.border}`,
    borderRadius: 12,
    padding: "20px 24px",
    ...style,
  }} className={className}>{children}</div>
);

const MetricPill = ({ label, value, sub, color = C.accent1 }) => (
  <div style={{
    background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10,
    padding: "14px 20px", display: "flex", flexDirection: "column", gap: 4,
    borderLeft: `3px solid ${color}`,
  }}>
    <span style={{ color: C.muted, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</span>
    <span style={{ color, fontSize: 26, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>{value}</span>
    {sub && <span style={{ color: C.muted, fontSize: 11 }}>{sub}</span>}
  </div>
);

const SectionTitle = ({ children, accent = C.accent1 }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
    <div style={{ width: 3, height: 20, background: accent, borderRadius: 2 }} />
    <span style={{ color: C.text, fontSize: 13, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>
      {children}
    </span>
  </div>
);

const HeatCell = ({ value }) => {
  const pct = value / 95;
  const r = Math.round(99 + pct * (139 - 99));
  const g = Math.round(102 + pct * (92 - 102));
  const b = Math.round(241 * (1 - pct * 0.6));
  return (
    <td style={{
      background: `rgba(${r},${g},${b},${0.2 + pct * 0.7})`,
      color: pct > 0.5 ? "#fff" : C.muted,
      textAlign: "center", padding: "10px 14px",
      fontSize: 13, fontFamily: "'Space Mono', monospace", fontWeight: 600,
      border: `1px solid ${C.border}`, borderRadius: 4, cursor: "default",
    }} title={`${value}%`}>
      {value}%
    </td>
  );
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#13152a", border: `1px solid ${C.border}`, borderRadius: 8,
      padding: "10px 14px", fontSize: 12,
    }}>
      <div style={{ color: C.muted, marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || C.text, marginBottom: 2 }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(1) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

// ─── TABS ─────────────────────────────────────────────────────────────────────

const TABS = ["Retention", "A/B Test", "Segmentation", "Monetization", "Funnel"];

// ─── MAIN DASHBOARD ──────────────────────────────────────────────────────────

export default function Dashboard() {
  const [tab, setTab] = useState("Retention");
  const [loaded, setLoaded] = useState(false);

  useEffect(() => { setTimeout(() => setLoaded(true), 200); }, []);

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      opacity: loaded ? 1 : 0, transition: "opacity 0.4s",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${C.border}`,
        padding: "18px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "linear-gradient(90deg, rgba(99,102,241,0.08) 0%, transparent 60%)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: "linear-gradient(135deg, #6366f1, #22d3ee)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>🎮</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>
              Mobile F2P Analytics — Homescapes-type
            </div>
            <div style={{ fontSize: 11, color: C.muted }}>
              Synthetic dataset · 50,000 players · Portfolio case study
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {[
            { label: "Players", val: `${(N/1000).toFixed(0)}K` },
            { label: "Cohorts", val: "6" },
            { label: "A/B Groups", val: "2" },
          ].map(m => (
            <div key={m.label} style={{
              background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: "6px 14px", textAlign: "center",
            }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: C.accent1, fontFamily: "'Space Mono', monospace" }}>{m.val}</div>
              <div style={{ fontSize: 10, color: C.muted }}>{m.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* KPI Strip */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 12, padding: "20px 32px 0" }}>
        <MetricPill label="D1 Retention" value="51.4%" sub="Overall avg" color={C.accent1} />
        <MetricPill label="D7 Retention" value="27.8%" sub="Overall avg" color={C.accent2} />
        <MetricPill label="D30 Retention" value="10.2%" sub="Modeled proxy" color={C.accent3} />
        <MetricPill label="A/B Uplift" value={`+${uplift}%`} sub={`p ${pValue}`} color={C.accent4} />
        <MetricPill label="Avg LTV (D30)" value="$14.2" sub="Blended" color={C.accent5} />
        <MetricPill label="Whale Share" value="5%" sub="62% of revenue" color={C.whale} />
      </div>

      {/* Tab Nav */}
      <div style={{ display: "flex", gap: 4, padding: "20px 32px 0" }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? C.accent1 : "transparent",
            color: tab === t ? "#fff" : C.muted,
            border: `1px solid ${tab === t ? C.accent1 : C.border}`,
            borderRadius: 8, padding: "7px 18px", fontSize: 12,
            fontWeight: 600, cursor: "pointer", letterSpacing: "0.03em",
            transition: "all 0.2s",
          }}>{t}</button>
        ))}
      </div>

      {/* Tab Content */}
      <div style={{ padding: "20px 32px 40px" }}>

        {/* ── RETENTION ── */}
        {tab === "Retention" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Retention Curves */}
            <Card style={{ gridColumn: "span 2" }}>
              <SectionTitle accent={C.accent1}>Retention Curves — Control vs Treatment (D0–D30)</SectionTitle>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={retentionCurve} margin={{ right: 20 }}>
                  <defs>
                    <linearGradient id="gc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.accent1} stopOpacity={0.25} />
                      <stop offset="95%" stopColor={C.accent1} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gt" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.accent2} stopOpacity={0.25} />
                      <stop offset="95%" stopColor={C.accent2} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="day" stroke={C.muted} tick={{ fontSize: 11 }} label={{ value: "Day", position: "insideBottomRight", fill: C.muted, fontSize: 11 }} />
                  <YAxis stroke={C.muted} tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <ReferenceLine x={1} stroke={C.muted} strokeDasharray="4 4" label={{ value: "D1", fill: C.muted, fontSize: 10 }} />
                  <ReferenceLine x={7} stroke={C.muted} strokeDasharray="4 4" label={{ value: "D7", fill: C.muted, fontSize: 10 }} />
                  <ReferenceLine x={30} stroke={C.muted} strokeDasharray="4 4" label={{ value: "D30", fill: C.muted, fontSize: 10 }} />
                  <Area type="monotone" dataKey="control" name="Control" stroke={C.accent1} fill="url(#gc)" strokeWidth={2} dot={false} />
                  <Area type="monotone" dataKey="treatment" name="Treatment" stroke={C.accent2} fill="url(#gt)" strokeWidth={2} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            {/* Cohort Heatmap */}
            <Card>
              <SectionTitle accent={C.accent3}>Cohort Retention Heatmap (%)</SectionTitle>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: 3 }}>
                  <thead>
                    <tr>
                      <th style={{ color: C.muted, fontSize: 11, textAlign: "left", paddingBottom: 8, fontWeight: 500 }}>Cohort</th>
                      {cohortWeeks.map(w => (
                        <th key={w} style={{ color: C.muted, fontSize: 11, textAlign: "center", fontWeight: 500 }}>{w}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cohortData.map(row => (
                      <tr key={row.cohort}>
                        <td style={{ color: C.muted, fontSize: 12, paddingRight: 12, paddingBottom: 3 }}>{row.cohort}</td>
                        {cohortWeeks.map(w => <HeatCell key={w} value={row[w]} />)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* Session Depth */}
            <Card>
              <SectionTitle accent={C.accent4}>Session Depth Distribution</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={sessionBuckets} margin={{ right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="sessions" stroke={C.muted} tick={{ fontSize: 11 }} label={{ value: "Sessions (first week)", position: "insideBottom", fill: C.muted, fontSize: 10, dy: 10 }} />
                  <YAxis stroke={C.muted} tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" name="Players" radius={[4, 4, 0, 0]}>
                    {sessionBuckets.map((_, i) => (
                      <Cell key={i} fill={`hsl(${240 + i * 18}, 80%, ${50 + i * 5}%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* ── A/B TEST ── */}
        {tab === "A/B Test" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Summary Cards */}
            <Card style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <SectionTitle accent={C.accent4}>A/B Test Summary</SectionTitle>
              <div style={{ gridColumn: "span 2", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  { label: "Control D7 Ret.", val: `${(ctrlMean * 100).toFixed(1)}%`, color: C.accent1 },
                  { label: "Treatment D7 Ret.", val: `${(treatMean * 100).toFixed(1)}%`, color: C.accent2 },
                  { label: "Relative Uplift", val: `+${uplift}%`, color: C.accent4 },
                  { label: "p-value", val: pValue, color: C.accent3 },
                  { label: "95% CI", val: `[+${ciLow}%, +${ciHigh}%]`, color: C.accent5 },
                  { label: "Z-score", val: zScore.toFixed(2), color: C.muted },
                  { label: "Sample (ctrl)", val: abControl.length.toLocaleString(), color: C.muted },
                  { label: "Sample (treat)", val: abTreat.length.toLocaleString(), color: C.muted },
                ].map(m => (
                  <div key={m.label} style={{
                    background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8,
                    padding: "12px 16px",
                  }}>
                    <div style={{ color: C.muted, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.07em" }}>{m.label}</div>
                    <div style={{ color: m.color, fontSize: 18, fontWeight: 700, fontFamily: "'Space Mono', monospace", marginTop: 4 }}>{m.val}</div>
                  </div>
                ))}
              </div>
            </Card>

            {/* Power Analysis */}
            <Card>
              <SectionTitle accent={C.accent2}>Statistical Power vs Sample Size</SectionTitle>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={powerCurve} margin={{ right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="n" stroke={C.muted} tick={{ fontSize: 10 }} tickFormatter={v => `${v / 1000}K`} label={{ value: "Sample size (per group)", position: "insideBottomRight", fill: C.muted, fontSize: 10 }} />
                  <YAxis stroke={C.muted} tick={{ fontSize: 10 }} unit="%" domain={[0, 100]} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={80} stroke={C.accent4} strokeDasharray="5 5" label={{ value: "80% power", fill: C.accent4, fontSize: 10, position: "right" }} />
                  <Line type="monotone" dataKey="power" name="Power" stroke={C.accent2} strokeWidth={2.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* A/B retention curve by day */}
            <Card style={{ gridColumn: "span 2" }}>
              <SectionTitle accent={C.accent1}>A/B Retention Uplift by Day</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={retentionCurve.filter(d => [1,3,7,14,21,30].includes(d.day))} margin={{ right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="day" stroke={C.muted} tick={{ fontSize: 11 }} tickFormatter={v => `D${v}`} />
                  <YAxis stroke={C.muted} tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="control" name="Control" fill={C.accent1} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="treatment" name="Treatment" fill={C.accent2} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* ── SEGMENTATION ── */}
        {tab === "Segmentation" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Segment Overview */}
            <Card>
              <SectionTitle accent={C.whale}>Segment Distribution & D7 Retention</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={segDist} layout="vertical" margin={{ right: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
                  <XAxis type="number" stroke={C.muted} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" stroke={C.muted} tick={{ fontSize: 12 }} width={70} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="d7Ret" name="D7 Retention %" radius={[0, 4, 4, 0]}>
                    {segDist.map(s => <Cell key={s.name} fill={SEG_COLORS[s.name]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Radar */}
            <Card>
              <SectionTitle accent={C.accent2}>Behavioral Profile Radar (per segment)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={[
                  { metric: "Retention", Whale: 58, Dolphin: 38, Minnow: 22, Ghost: 6 },
                  { metric: "Sessions", Whale: 90, Dolphin: 55, Minnow: 28, Ghost: 8 },
                  { metric: "Spend", Whale: 95, Dolphin: 40, Minnow: 8, Ghost: 2 },
                  { metric: "Levels", Whale: 80, Dolphin: 50, Minnow: 22, Ghost: 5 },
                  { metric: "D30 Ret", Whale: 64, Dolphin: 32, Minnow: 14, Ghost: 2 },
                ]}>
                  <PolarGrid stroke={C.border} />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: C.muted, fontSize: 11 }} />
                  {SEGMENTS.map(s => (
                    <Radar key={s} name={s} dataKey={s} stroke={SEG_COLORS[s]} fill={SEG_COLORS[s]} fillOpacity={0.12} strokeWidth={2} />
                  ))}
                  <Legend />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>

            {/* Segment table */}
            <Card style={{ gridColumn: "span 2" }}>
              <SectionTitle accent={C.accent3}>Segment KPI Breakdown</SectionTitle>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                    {["Segment", "Players", "Share", "D7 Ret", "Avg Spend", "Revenue Share"].map(h => (
                      <th key={h} style={{ color: C.muted, fontWeight: 500, padding: "8px 16px", textAlign: "left", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {segDist.map((s, i) => {
                    const segPlayers = players.filter(p => p.seg === s.name);
                    const totalRevenue = players.reduce((a, p) => a + p.spend, 0);
                    const segRevenue = segPlayers.reduce((a, p) => a + p.spend, 0);
                    return (
                      <tr key={s.name} style={{ borderBottom: `1px solid ${C.border}`, background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent" }}>
                        <td style={{ padding: "12px 16px" }}>
                          <span style={{ color: SEG_COLORS[s.name], fontWeight: 700 }}>● </span>
                          <span style={{ color: C.text }}>{s.name}</span>
                        </td>
                        <td style={{ padding: "12px 16px", fontFamily: "'Space Mono', monospace", color: C.text }}>{s.count.toLocaleString()}</td>
                        <td style={{ padding: "12px 16px", color: C.muted }}>{(s.count / N * 100).toFixed(0)}%</td>
                        <td style={{ padding: "12px 16px", color: SEG_COLORS[s.name], fontFamily: "'Space Mono', monospace", fontWeight: 600 }}>{s.d7Ret}%</td>
                        <td style={{ padding: "12px 16px", fontFamily: "'Space Mono', monospace", color: C.text }}>${s.avgLTV}</td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div style={{ flex: 1, height: 6, background: C.border, borderRadius: 3, overflow: "hidden" }}>
                              <div style={{ width: `${segRevenue / totalRevenue * 100}%`, height: "100%", background: SEG_COLORS[s.name], borderRadius: 3 }} />
                            </div>
                            <span style={{ fontFamily: "'Space Mono', monospace", fontSize: 12, color: C.text, minWidth: 38 }}>
                              {(segRevenue / totalRevenue * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Card>
          </div>
        )}

        {/* ── MONETIZATION ── */}
        {tab === "Monetization" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* LTV Model */}
            <Card style={{ gridColumn: "span 2" }}>
              <SectionTitle accent={C.accent5}>LTV Proxy Model — Predicted vs Actual (30-day)</SectionTitle>
              <div style={{ display: "flex", gap: 32, alignItems: "flex-start" }}>
                <ResponsiveContainer width="55%" height={260}>
                  <BarChart data={ltvModelData} margin={{ right: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                    <XAxis dataKey="segment" stroke={C.muted} tick={{ fontSize: 12 }} />
                    <YAxis stroke={C.muted} tick={{ fontSize: 11 }} unit="$" />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Bar dataKey="actual" name="Actual LTV" fill={C.accent1} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="predicted" name="Predicted LTV" fill={C.accent5} radius={[4, 4, 0, 0]} opacity={0.8} />
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ flex: 1 }}>
                  <div style={{ color: C.muted, fontSize: 12, lineHeight: 1.8 }}>
                    <p style={{ color: C.text, fontWeight: 600, marginBottom: 12 }}>Model Notes</p>
                    <p>• LTV proxy uses D1/D7 retention + session depth as early signals</p>
                    <p>• Whale segment predicted within <span style={{ color: C.accent4 }}>±8%</span> of actual</p>
                    <p>• Ghost segment has high uncertainty due to near-zero engagement</p>
                    <p>• Model RMSE: <span style={{ color: C.accent2, fontFamily: "monospace" }}>$4.2</span> blended across segments</p>
                    <p style={{ marginTop: 12 }}>Early signal correlations:</p>
                    <p>• D1 session count → D30 spend: <span style={{ color: C.accent5 }}>r = 0.61</span></p>
                    <p>• Levels cleared D1-3 → D30 LTV: <span style={{ color: C.accent5 }}>r = 0.74</span></p>
                  </div>
                </div>
              </div>
            </Card>

            {/* Revenue by segment (scatter) */}
            <Card style={{ gridColumn: "span 2" }}>
              <SectionTitle accent={C.accent3}>Spend Distribution by Segment (sample 800 players)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart margin={{ right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis type="number" dataKey="sessions" name="Sessions" stroke={C.muted} tick={{ fontSize: 11 }} label={{ value: "Sessions (week 1)", position: "insideBottomRight", fill: C.muted, fontSize: 10 }} />
                  <YAxis type="number" dataKey="spend" name="Spend ($)" stroke={C.muted} tick={{ fontSize: 11 }} unit="$" />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0]?.payload;
                    return (
                      <div style={{ background: "#13152a", border: `1px solid ${C.border}`, borderRadius: 8, padding: "8px 12px", fontSize: 12 }}>
                        <div style={{ color: SEG_COLORS[d?.seg] }}>{d?.seg}</div>
                        <div style={{ color: C.text }}>Sessions: {d?.sessions}</div>
                        <div style={{ color: C.text }}>Spend: ${d?.spend?.toFixed(1)}</div>
                      </div>
                    );
                  }} />
                  <Legend />
                  {SEGMENTS.map(s => (
                    <Scatter
                      key={s} name={s}
                      data={players.filter(p => p.seg === s && p.spend > 0).slice(0, 200).map(p => ({ sessions: p.sessions, spend: p.spend, seg: p.seg }))}
                      fill={SEG_COLORS[s]} opacity={0.7}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* ── FUNNEL ── */}
        {tab === "Funnel" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* Level Funnel */}
            <Card>
              <SectionTitle accent={C.accent1}>Level Progression Funnel</SectionTitle>
              <div style={{ marginTop: 8 }}>
                {levelFunnel.map((l, i) => {
                  const pct = (l.users / N * 100).toFixed(0);
                  const dropPct = i > 0 ? (100 - l.users / levelFunnel[i - 1].users * 100).toFixed(0) : null;
                  return (
                    <div key={l.level} style={{ marginBottom: 10 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ color: C.text, fontSize: 13, fontWeight: 600 }}>{l.level}</span>
                        <span style={{ color: C.muted, fontSize: 12 }}>
                          {l.users.toLocaleString()} players
                          {dropPct && <span style={{ color: C.accent3, marginLeft: 8 }}>↓ {dropPct}% drop</span>}
                        </span>
                      </div>
                      <div style={{ height: 32, background: C.border, borderRadius: 6, overflow: "hidden", position: "relative" }}>
                        <div style={{
                          width: `${pct}%`, height: "100%",
                          background: `linear-gradient(90deg, ${l.color}, ${l.color}aa)`,
                          borderRadius: 6,
                          display: "flex", alignItems: "center", paddingLeft: 10,
                          transition: "width 0.6s ease",
                        }}>
                          <span style={{ color: "#fff", fontSize: 12, fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>{pct}%</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>

            {/* Churn signals */}
            <Card>
              <SectionTitle accent={C.accent3}>Churn Risk Signals (Feature Importance)</SectionTitle>
              {[
                { feature: "Sessions D1-3 < 2", importance: 0.31, color: C.accent3 },
                { feature: "No progression D1", importance: 0.24, color: C.accent5 },
                { feature: "D1 ret. < 15 min", importance: 0.18, color: C.accent1 },
                { feature: "Skipped tutorial", importance: 0.13, color: C.accent2 },
                { feature: "0 levels D1", importance: 0.09, color: C.muted },
                { feature: "No push opt-in", importance: 0.05, color: C.muted },
              ].map(f => (
                <div key={f.feature} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 12, color: C.text }}>{f.feature}</span>
                    <span style={{ fontSize: 11, color: f.color, fontFamily: "'Space Mono', monospace" }}>{(f.importance * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{ height: 8, background: C.border, borderRadius: 4, overflow: "hidden" }}>
                    <div style={{ width: `${f.importance * 100 / 0.31 * 100}%`, height: "100%", background: f.color, borderRadius: 4 }} />
                  </div>
                </div>
              ))}
              <div style={{ marginTop: 16, padding: "12px 16px", background: C.bg, borderRadius: 8, border: `1px solid ${C.border}` }}>
                <div style={{ color: C.muted, fontSize: 11, lineHeight: 1.7 }}>
                  <span style={{ color: C.accent4, fontWeight: 600 }}>Key insight:</span> Players with &lt;2 sessions in first 3 days have <span style={{ color: C.accent3 }}>4.2× higher churn</span> probability. Recommend push notification trigger at 36h inactivity for new users.
                </div>
              </div>
            </Card>

            {/* Guardrail metrics */}
            <Card style={{ gridColumn: "span 2" }}>
              <SectionTitle accent={C.accent4}>A/B Guardrail Metrics — Long-term Risk Assessment</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                {[
                  { metric: "Session length", control: "8.2 min", treatment: "8.5 min", delta: "+3.7%", status: "✓ Safe" },
                  { metric: "Crash rate", control: "0.8%", treatment: "0.9%", delta: "+0.1pp", status: "✓ Safe" },
                  { metric: "IAP conversion", control: "4.1%", treatment: "4.8%", delta: "+17%", status: "✓ Uplift" },
                  { metric: "Support tickets", control: "1.2%", treatment: "1.3%", delta: "+0.1pp", status: "⚠ Monitor" },
                ].map(g => (
                  <div key={g.metric} style={{
                    background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 16px",
                  }}>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 8 }}>{g.metric}</div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ color: C.muted, fontSize: 11 }}>Ctrl:</span>
                      <span style={{ color: C.text, fontSize: 12, fontFamily: "monospace" }}>{g.control}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <span style={{ color: C.muted, fontSize: 11 }}>Treat:</span>
                      <span style={{ color: C.text, fontSize: 12, fontFamily: "monospace" }}>{g.treatment}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ color: C.accent4, fontSize: 12, fontFamily: "monospace" }}>{g.delta}</span>
                      <span style={{
                        fontSize: 10, padding: "2px 8px", borderRadius: 4,
                        background: g.status.startsWith("✓") ? "rgba(52,211,153,0.12)" : "rgba(251,146,60,0.12)",
                        color: g.status.startsWith("✓") ? C.accent4 : C.accent5,
                      }}>{g.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: `1px solid ${C.border}`, padding: "14px 32px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span style={{ color: C.muted, fontSize: 11 }}>
          Synthetic dataset · 50,000 players · Homescapes-type F2P · Portfolio: github.com/Gobberz
        </span>
        <span style={{ color: C.muted, fontSize: 11 }}>Python · SQL · Statistical Modeling · 2026</span>
      </div>
    </div>
  );
}
