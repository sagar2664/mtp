"""
Premium Streamlit Dashboard for Green & Resilient Supply Chain Network.
Provides interactive visualization of Pareto-optimal solutions, network flows,
simulation results, and KPI analysis.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "./data"
RESULTS_DIR = "./results"

# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0e1117",
    "card":        "#1a1d23",
    "card_border": "#2d333b",
    "primary":     "#58a6ff",
    "green":       "#3fb950",
    "orange":      "#d29922",
    "red":         "#f85149",
    "purple":      "#bc8cff",
    "text":        "#c9d1d9",
    "text_dim":    "#8b949e",
    "road":        "#f97316",
    "rail":        "#3b82f6",
    "supplier":    "#22c55e",
    "factory":     "#3b82f6",
    "dc":          "#f59e0b",
    "customer":    "#ef4444",
}

PLOTLY_TEMPLATE = "plotly_dark"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Green Resilient SCN – Decision Support",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"], .stApp {
    font-family: 'Inter', sans-serif;
}
.stApp { background-color: #0e1117; }

/* KPI metric cards */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi {
    flex: 1;
    background: linear-gradient(135deg, #1a1d23 0%, #21262d 100%);
    border: 1px solid #2d333b;
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(88,166,255,0.12);
}
.kpi-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: #8b949e; margin-bottom: 6px; }
.kpi-value { font-size: 1.75rem; font-weight: 800; background: linear-gradient(90deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi-sub { font-size: 0.72rem; color: #6e7681; margin-top: 4px; }

/* Section headers */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #c9d1d9;
    border-bottom: 2px solid #58a6ff; padding-bottom: 8px;
    margin: 28px 0 16px 0; letter-spacing: 0.5px;
}

/* Legend chips */
.chip-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 18px 0; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #21262d; border: 1px solid #30363d; border-radius: 20px;
    padding: 5px 14px; font-size: 0.75rem; color: #c9d1d9;
}
.chip-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stRadio > label { font-weight: 600; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #21262d; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 8px 20px;
    font-weight: 600; font-size: 0.85rem;
    background: transparent; color: #8b949e;
}
.stTabs [aria-selected="true"] { background: #21262d; color: #58a6ff; border-bottom: 2px solid #58a6ff; }

/* Hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        suppliers = pd.read_csv(f"{DATA_DIR}/suppliers.csv")
        factories = pd.read_csv(f"{DATA_DIR}/factories.csv")
        dcs       = pd.read_csv(f"{DATA_DIR}/dcs.csv")
        customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
    except Exception as e:
        st.error(f"❌ Node data not found – run `python results/run_experiment.py` first.\n\n{e}")
        return [None]*6
    try:
        pareto  = pd.read_csv(f"{RESULTS_DIR}/pareto_front.csv")
        with open(f"{RESULTS_DIR}/experiment_summary.json") as f:
            summary = json.load(f)
    except Exception as e:
        st.error(f"❌ Results data not found – run the pipeline first.\n\n{e}")
        return suppliers, factories, dcs, customers, None, None
    return suppliers, factories, dcs, customers, pareto, summary


@st.cache_data
def load_flows(label):
    tag = label.replace(' ', '_').lower()
    dfs = []
    for prefix in ['flows_supplier_factory', 'flows_factory_dc', 'flows_dc_customer']:
        path = f"{RESULTS_DIR}/{prefix}_{tag}.csv"
        dfs.append(pd.read_csv(path) if os.path.exists(path) else pd.DataFrame())
    return tuple(dfs)


@st.cache_data
def load_demand():
    path = f"{DATA_DIR}/demand.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ── Chart builders ────────────────────────────────────────────────────────────
def _plotly_layout(fig, height=500, margin=None):
    """Apply consistent dark theme to any plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161b22",
        font=dict(family="Inter", color="#c9d1d9"),
        height=height,
        margin=margin or dict(l=48, r=24, t=48, b=40),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor="#30363d", borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig


def build_pareto_3d(pareto_df, reps):
    """3-D Pareto front with representative solutions highlighted."""
    fig = go.Figure()

    # All Pareto points
    fig.add_trace(go.Scatter3d(
        x=pareto_df['cost'], y=pareto_df['emissions'], z=pareto_df['resilience'],
        mode='markers',
        marker=dict(
            size=4, color=pareto_df['resilience'],
            colorscale='Viridis', opacity=0.55,
            colorbar=dict(title="Resilience", thickness=14, len=0.6),
            line=dict(width=0),
        ),
        name='Pareto Front',
        hovertemplate="Cost: $%{x:,.0f}<br>Emissions: %{y:,.0f} kg<br>Resilience: %{z:.1%}<extra></extra>",
    ))

    # Highlight representative solutions
    marker_symbols = ['diamond', 'cross', 'circle', 'square']
    rep_colors = [COLORS['green'], COLORS['primary'], COLORS['purple'], COLORS['orange']]
    for i, r in enumerate(reps):
        fig.add_trace(go.Scatter3d(
            x=[r['cost']], y=[r['emissions']], z=[r['resilience']],
            mode='markers+text',
            marker=dict(size=9, color=rep_colors[i % len(rep_colors)], symbol=marker_symbols[i % 4],
                        line=dict(width=2, color='white')),
            text=[r['label']], textposition='top center',
            textfont=dict(size=10, color=rep_colors[i % len(rep_colors)]),
            name=r['label'],
            hovertemplate=f"<b>{r['label']}</b><br>Cost: ${r['cost']:,.0f}<br>Emissions: {r['emissions']:,.0f} kg<br>Resilience: {r['resilience']:.1%}<extra></extra>",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Total Cost ($)", backgroundcolor="#161b22", gridcolor="#21262d"),
            yaxis=dict(title="Emissions (kg CO₂)", backgroundcolor="#161b22", gridcolor="#21262d"),
            zaxis=dict(title="Resilience", backgroundcolor="#161b22", gridcolor="#21262d"),
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=36, b=0),
        height=560,
        font=dict(family="Inter", color="#c9d1d9"),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
    )
    return fig


def build_pareto_2d(pareto_df, reps):
    """2-D projection subplots with representative solutions."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        "Cost vs Emissions", "Cost vs Resilience", "Emissions vs Resilience"
    ], horizontal_spacing=0.07)

    pairs = [('cost','emissions','resilience'), ('cost','resilience','emissions'), ('emissions','resilience','cost')]
    for idx, (xc, yc, cc) in enumerate(pairs):
        col = idx + 1
        fig.add_trace(go.Scatter(
            x=pareto_df[xc], y=pareto_df[yc], mode='markers',
            marker=dict(size=5, color=pareto_df[cc], colorscale='Viridis', opacity=0.6,
                        showscale=(col == 3),
                        colorbar=dict(title=cc.capitalize(), thickness=12, len=0.8) if col == 3 else None),
            showlegend=False,
            hovertemplate=f"{xc.capitalize()}: %{{x:,.0f}}<br>{yc.capitalize()}: %{{y:,.4f}}<extra></extra>",
        ), row=1, col=col)
        # Representative dots
        rep_colors_list = [COLORS['green'], COLORS['primary'], COLORS['purple'], COLORS['orange']]
        for i, r in enumerate(reps):
            fig.add_trace(go.Scatter(
                x=[r[xc]], y=[r[yc]], mode='markers+text',
                marker=dict(size=11, color=rep_colors_list[i%4], line=dict(width=2, color='white')),
                text=[r['label'].split('(')[0].strip()], textposition='top center',
                textfont=dict(size=9, color=rep_colors_list[i%4]),
                showlegend=(col == 1),
                name=r['label'] if col == 1 else None,
            ), row=1, col=col)

    _plotly_layout(fig, height=380, margin=dict(l=56, r=24, t=48, b=44))
    return fig


def build_network_map(suppliers, factories, dcs, customers, sf_flows, fd_flows, dc_flows):
    """Geographic network map of India with mode-colored flow arcs."""
    fig = go.Figure()

    # Nodes
    node_defs = [
        (suppliers, 'supplier_id', 'Suppliers',  COLORS['supplier'], 'square',       15),
        (factories, 'factory_id', 'Factories',   COLORS['factory'],  'triangle-up',  15),
        (dcs,       'dc_id',      'DCs',         COLORS['dc'],       'diamond',       13),
        (customers, 'customer_id','Customers',   COLORS['customer'], 'circle',        9),
    ]
    for df, id_col, name, color, symbol, sz in node_defs:
        fig.add_trace(go.Scattergeo(
            lon=df['longitude'], lat=df['latitude'],
            mode='markers+text',
            marker=dict(size=sz, color=color, symbol=symbol, line=dict(width=1.5, color='white')),
            text=df[id_col], textposition='top center',
            textfont=dict(size=10, color='white', family='Inter'),
            name=name,
            hovertemplate=f"<b>%{{text}}</b><br>Lat: %{{lat:.2f}}, Lon: %{{lon:.2f}}<extra>{name}</extra>",
        ))

    # Flow arcs
    def _add_arcs(flow_df, src_df, tgt_df, src_col, tgt_col):
        if flow_df.empty:
            return
        max_flow = flow_df['flow'].max() if 'flow' in flow_df.columns else 1
        for _, row in flow_df.iterrows():
            try:
                src_key = src_col.replace('_id', '')
                tgt_key = tgt_col.replace('_id', '')
                s = src_df[src_df[src_col] == row[src_key]].iloc[0]
                t = tgt_df[tgt_df[tgt_col] == row[tgt_key]].iloc[0]
                m = row.get('mode', 'road')
                w = max(1.2, (row['flow'] / max_flow) * 5) if max_flow > 0 else 1.2
                fig.add_trace(go.Scattergeo(
                    lon=[s['longitude'], t['longitude']],
                    lat=[s['latitude'], t['latitude']],
                    mode='lines',
                    line=dict(width=w, color=COLORS.get(m, '#666')),
                    opacity=0.7 if m == 'road' else 0.85,
                    showlegend=False,
                    hoverinfo='text',
                    text=f"<b>{row[src_key]} → {row[tgt_key]}</b><br>Mode: {m.title()}<br>Flow: {row['flow']:.1f} units",
                ))
            except Exception:
                pass

    _add_arcs(sf_flows, suppliers, factories, 'supplier_id', 'factory_id')
    _add_arcs(fd_flows, factories, dcs, 'factory_id', 'dc_id')
    _add_arcs(dc_flows, dcs, customers, 'dc_id', 'customer_id')

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9"),
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
        margin=dict(l=0, r=0, t=8, b=0),
        height=620,
        geo=dict(
            scope='asia',
            projection_type='mercator',
            showland=True,  landcolor='#1a1d23',
            showocean=True, oceancolor='#0e1117',
            showcountries=True, countrycolor='#30363d',
            showlakes=True, lakecolor='#0e1117',
            showrivers=False,
            center=dict(lon=80, lat=22),
            projection_scale=4.5,
            bgcolor='#0e1117',
        ),
    )
    return fig


def build_solution_radar(reps):
    """Radar chart comparing representative solutions."""
    categories = ['Cost Efficiency', 'Emission Efficiency', 'Resilience']

    costs = [r['cost'] for r in reps]
    emiss = [r['emissions'] for r in reps]
    resil = [r['resilience'] for r in reps]

    # Normalize: cost & emission are "lower is better" → invert; resilience is "higher is better"
    ce = [1 - (c - min(costs)) / (max(costs) - min(costs) + 1e-9) for c in costs]
    ee = [1 - (e - min(emiss)) / (max(emiss) - min(emiss) + 1e-9) for e in emiss]
    rn = [(r - min(resil)) / (max(resil) - min(resil) + 1e-9) for r in resil]

    rep_colors = [COLORS['green'], COLORS['primary'], COLORS['purple'], COLORS['orange']]

    fig = go.Figure()
    for i, r in enumerate(reps):
        vals = [ce[i], ee[i], rn[i]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=rep_colors[i % 4].replace(')', ',0.08)').replace('rgb', 'rgba') if 'rgb' in rep_colors[i%4] else f"rgba({int(rep_colors[i%4][1:3],16)},{int(rep_colors[i%4][3:5],16)},{int(rep_colors[i%4][5:7],16)},0.08)",
            line=dict(color=rep_colors[i % 4], width=2),
            name=r['label'],
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#21262d', color='#8b949e'),
            angularaxis=dict(gridcolor='#21262d', color='#c9d1d9', linecolor='#30363d'),
        ),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9"),
        height=420,
        margin=dict(l=60, r=60, t=36, b=36),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
    )
    return fig


def build_solution_bar(reps):
    """Grouped bar chart for representative solutions."""
    labels = [r['label'] for r in reps]
    costs  = [r['cost'] for r in reps]
    emiss  = [r['emissions'] for r in reps]
    resil  = [r['resilience'] * 100 for r in reps]

    # Normalize each to 0-100 for comparison
    def norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-9) * 100 for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Cost (norm)', x=labels, y=norm(costs),
                         marker_color=COLORS['red'], marker_line_width=0, opacity=0.85))
    fig.add_trace(go.Bar(name='Emissions (norm)', x=labels, y=norm(emiss),
                         marker_color=COLORS['orange'], marker_line_width=0, opacity=0.85))
    fig.add_trace(go.Bar(name='Resilience (%)', x=labels, y=resil,
                         marker_color=COLORS['green'], marker_line_width=0, opacity=0.85))

    fig.update_layout(barmode='group', bargap=0.2, bargroupgap=0.06)
    _plotly_layout(fig, height=400)
    fig.update_layout(xaxis_title=None, yaxis_title="Normalized Score / %")
    return fig


def build_mode_sunburst(sf, fd, dc):
    """Sunburst chart for transport mode share by stage."""
    rows = []
    stage_map = [(sf, 'S→F'), (fd, 'F→DC'), (dc, 'DC→C')]
    for df, stage in stage_map:
        if df.empty:
            continue
        for mode in ['road', 'rail']:
            total = df[df['mode'] == mode]['flow'].sum() if 'mode' in df.columns else 0
            if total > 0:
                rows.append(dict(stage=stage, mode=mode.title(), flow=total))

    if not rows:
        return None

    burst_df = pd.DataFrame(rows)
    fig = px.sunburst(burst_df, path=['stage', 'mode'], values='flow',
                      color='mode',
                      color_discrete_map={'Road': COLORS['road'], 'Rail': COLORS['rail']})
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9"),
        height=400, margin=dict(l=8, r=8, t=36, b=8),
    )
    fig.update_traces(textinfo='label+percent parent', insidetextorientation='radial')
    return fig


def build_flow_sankey(sf, fd, dc):
    """Sankey diagram showing flow through the network."""
    all_nodes = []
    node_map = {}

    def get_idx(name):
        if name not in node_map:
            node_map[name] = len(all_nodes)
            all_nodes.append(name)
        return node_map[name]

    sources, targets, values, link_colors = [], [], [], []
    color_map = {'road': 'rgba(249,115,22,0.35)', 'rail': 'rgba(59,130,246,0.35)'}

    for df, src_col, tgt_col in [(sf,'supplier','factory'), (fd,'factory','dc'), (dc,'dc','customer')]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            s = get_idx(row[src_col])
            t = get_idx(row[tgt_col])
            sources.append(s)
            targets.append(t)
            values.append(row['flow'])
            link_colors.append(color_map.get(row.get('mode','road'), 'rgba(100,100,100,0.3)'))

    if not sources:
        return None

    # Node colors by type
    node_colors = []
    for n in all_nodes:
        if n.startswith('S'): node_colors.append(COLORS['supplier'])
        elif n.startswith('F'): node_colors.append(COLORS['factory'])
        elif n.startswith('D'): node_colors.append(COLORS['dc'])
        else: node_colors.append(COLORS['customer'])

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(pad=20, thickness=18, line=dict(color='#30363d', width=1),
                  label=all_nodes, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c9d1d9", size=12),
        height=450, margin=dict(l=24, r=24, t=36, b=24),
    )
    return fig


def build_simulation_gauges(sim):
    """Gauge charts for simulation KPIs."""
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{"type":"indicator"},{"type":"indicator"},{"type":"indicator"}]],
                        horizontal_spacing=0.08)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sim['average_fill_rate'] * 100,
        number=dict(suffix='%', font=dict(size=32, color=COLORS['green'])),
        title=dict(text="Avg Fill Rate", font=dict(size=13, color='#8b949e')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#30363d'),
            bar=dict(color=COLORS['green']),
            bgcolor='#21262d', borderwidth=0,
            steps=[dict(range=[0,60], color='#3d1f1f'), dict(range=[60,80], color='#3d3d1f'), dict(range=[80,100], color='#1f3d1f')],
        ),
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sim['min_fill_rate'] * 100,
        number=dict(suffix='%', font=dict(size=32, color=COLORS['orange'])),
        title=dict(text="Worst-Case Fill Rate", font=dict(size=13, color='#8b949e')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#30363d'),
            bar=dict(color=COLORS['orange']),
            bgcolor='#21262d', borderwidth=0,
            steps=[dict(range=[0,50], color='#3d1f1f'), dict(range=[50,75], color='#3d3d1f'), dict(range=[75,100], color='#1f3d1f')],
        ),
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sim['average_n_disruptions'],
        number=dict(font=dict(size=32, color=COLORS['red'])),
        title=dict(text="Avg Disruptions / 10yr", font=dict(size=13, color='#8b949e')),
        gauge=dict(
            axis=dict(range=[0, 10], tickcolor='#30363d'),
            bar=dict(color=COLORS['red']),
            bgcolor='#21262d', borderwidth=0,
        ),
    ), row=1, col=3)

    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"), height=300,
        margin=dict(l=32, r=32, t=48, b=16),
    )
    return fig


def build_demand_chart(demand_df):
    """Time-series demand chart for all customers."""
    if demand_df.empty:
        return None
    fig = px.line(demand_df, x='period', y='demand', color='customer_id',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    _plotly_layout(fig, height=380)
    fig.update_layout(
        xaxis_title='Period', yaxis_title='Demand (units)',
        legend_title_text='Customer',
    )
    return fig


# ── KPI renderer ──────────────────────────────────────────────────────────────
def render_kpis(sol, sim=None, forecast=None):
    # Row 1: Three core objectives
    row1 = f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="kpi-label">💰 Total Cost</div>
            <div class="kpi-value">${sol['cost']:,.0f}</div>
            <div class="kpi-sub">Melo et al. (2009) formulation</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">🌿 Carbon Emissions</div>
            <div class="kpi-value">{sol['emissions']:,.0f} kg</div>
            <div class="kpi-sub">CO₂ – Pishvaee &amp; Razmi (2012)</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">🛡️ Resilience Score</div>
            <div class="kpi-value">{sol['resilience']*100:.1f}%</div>
            <div class="kpi-sub">HHI + Node Failure composite</div>
        </div>
    </div>
    """
    st.markdown(row1, unsafe_allow_html=True)

    # Row 2: Simulation + Forecast (native Streamlit metrics)
    if sim or forecast:
        cols = st.columns(3)
        if sim:
            with cols[0]:
                st.metric("📊 Avg Fill Rate", f"{sim['average_fill_rate']*100:.1f}%", help="100 Monte Carlo runs")
            with cols[1]:
                st.metric("⚠️ Worst-Case Fill", f"{sim['min_fill_rate']*100:.1f}%", delta=f"{sim['average_n_disruptions']:.1f} avg disruptions", delta_color="inverse")
        if forecast:
            with cols[2]:
                st.metric("🎯 Forecast MAPE", f"{forecast['MAPE']:.1f}%", help="Random Forest accuracy")


def render_legend():
    st.markdown("""
    <div class="chip-row">
        <div class="chip"><span class="chip-dot" style="background:#22c55e;"></span> Suppliers</div>
        <div class="chip"><span class="chip-dot" style="background:#3b82f6;"></span> Factories</div>
        <div class="chip"><span class="chip-dot" style="background:#f59e0b;"></span> DCs</div>
        <div class="chip"><span class="chip-dot" style="background:#ef4444;"></span> Customers</div>
        <div class="chip"><span class="chip-dot" style="background:#f97316;"></span> Road</div>
        <div class="chip"><span class="chip-dot" style="background:#3b82f6;"></span> Rail</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    suppliers, factories, dcs, customers, pareto, summary = load_data()
    if summary is None:
        st.stop()

    reps = summary.get('representative_solutions', [])
    rep_labels = [r['label'] for r in reps]
    sim = summary.get('simulation_results', {})
    forecast = summary.get('forecast_accuracy', {})

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌿 SCN Dashboard")
        st.caption("Multi-Objective Supply Chain\nNetwork Design — MTP-2")
        st.markdown("---")

        page = st.radio("Navigate", [
            "🏠 Overview",
            "🗺️ Network Map",
            "📈 Pareto Analysis",
            "🔬 Solution Deep-Dive",
            "🛡️ Resilience Sim",
        ], label_visibility="collapsed")

        st.markdown("---")
        selected_label = st.selectbox("🔧 Active Solution", rep_labels)
        sel = next((r for r in reps if r['label'] == selected_label), reps[0])

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.72rem; color:#6e7681; line-height:1.6;">
            <b>Pipeline Info</b><br>
            Algorithm: NSGA-II<br>
            Pareto solutions: {summary['optimization']['n_pareto_solutions']}<br>
            Exec time: {summary['optimization']['execution_time_seconds']:.1f}s<br>
            Network: {summary['network_structure']['n_suppliers']}S–{summary['network_structure']['n_factories']}F–{summary['network_structure']['n_dcs']}D–{summary['network_structure']['n_customers']}C
        </div>
        """, unsafe_allow_html=True)

    sf_flows, fd_flows, dc_flows = load_flows(selected_label)

    # ═══════════════════════════════════════════════════════════════════════
    if page == "🏠 Overview":
        st.markdown("# 🌿 Green & Resilient Supply Chain — Overview")
        render_kpis(sel, sim, forecast)
        render_legend()

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown('<div class="section-header">Logistics Network</div>', unsafe_allow_html=True)
            fig_map = build_network_map(suppliers, factories, dcs, customers, sf_flows, fd_flows, dc_flows)
            st.plotly_chart(fig_map, key="overview_map", use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">Trade-off Space (3D)</div>', unsafe_allow_html=True)
            fig_3d = build_pareto_3d(pareto, reps)
            fig_3d.update_layout(height=480)
            st.plotly_chart(fig_3d, key="overview_3d", use_container_width=True)

            st.markdown('<div class="section-header">Solution Comparison</div>', unsafe_allow_html=True)
            fig_radar = build_solution_radar(reps)
            fig_radar.update_layout(height=350)
            st.plotly_chart(fig_radar, key="overview_radar", use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    elif page == "🗺️ Network Map":
        st.markdown(f"# 🗺️ Network Map — {selected_label}")
        render_kpis(sel)
        render_legend()

        fig_map = build_network_map(suppliers, factories, dcs, customers, sf_flows, fd_flows, dc_flows)
        fig_map.update_layout(height=700)
        st.plotly_chart(fig_map, key="map_full", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Flow Sankey Diagram</div>', unsafe_allow_html=True)
            sankey = build_flow_sankey(sf_flows, fd_flows, dc_flows)
            if sankey:
                st.plotly_chart(sankey, key="sankey", use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Mode Share by Stage</div>', unsafe_allow_html=True)
            sunburst = build_mode_sunburst(sf_flows, fd_flows, dc_flows)
            if sunburst:
                st.plotly_chart(sunburst, key="sunburst", use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    elif page == "📈 Pareto Analysis":
        st.markdown("# 📈 Multi-Objective Pareto Analysis")
        st.caption(f"{len(pareto)} non-dominated solutions from NSGA-II")

        fig_3d = build_pareto_3d(pareto, reps)
        fig_3d.update_layout(height=620)
        st.plotly_chart(fig_3d, key="pareto_3d_full", use_container_width=True)

        st.markdown('<div class="section-header">2-D Projections</div>', unsafe_allow_html=True)
        fig_2d = build_pareto_2d(pareto, reps)
        st.plotly_chart(fig_2d, key="pareto_2d", use_container_width=True)

        st.markdown('<div class="section-header">Pareto Front Data</div>', unsafe_allow_html=True)
        display_df = pareto.copy()
        display_df['cost'] = display_df['cost'].map('${:,.0f}'.format)
        display_df['emissions'] = display_df['emissions'].map('{:,.0f} kg'.format)
        display_df['resilience'] = display_df['resilience'].map('{:.1%}'.format)
        st.dataframe(display_df, use_container_width=True, height=300)

    # ═══════════════════════════════════════════════════════════════════════
    elif page == "🔬 Solution Deep-Dive":
        st.markdown("# 🔬 Solution Deep-Dive")

        tab_compare, tab_demand = st.tabs(["📊 Compare Solutions", "📉 Demand Forecast"])

        with tab_compare:
            render_kpis(sel)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-header">Radar Comparison</div>', unsafe_allow_html=True)
                st.plotly_chart(build_solution_radar(reps), key="deep_radar", use_container_width=True)
            with c2:
                st.markdown('<div class="section-header">Bar Comparison</div>', unsafe_allow_html=True)
                st.plotly_chart(build_solution_bar(reps), key="deep_bar", use_container_width=True)

            st.markdown('<div class="section-header">Solution Table</div>', unsafe_allow_html=True)
            table_data = []
            for r in reps:
                table_data.append({
                    'Solution': r['label'],
                    'Cost ($)': f"${r['cost']:,.0f}",
                    'Emissions (kg CO₂)': f"{r['emissions']:,.0f}",
                    'Resilience': f"{r['resilience']:.1%}",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        with tab_demand:
            demand_df = load_demand()
            if not demand_df.empty:
                st.markdown('<div class="section-header">Historical Demand</div>', unsafe_allow_html=True)
                fig_dem = build_demand_chart(demand_df)
                if fig_dem:
                    st.plotly_chart(fig_dem, key="demand_ts", use_container_width=True)
            else:
                st.info("Demand data not found in data/ directory.")

    # ═══════════════════════════════════════════════════════════════════════
    elif page == "🛡️ Resilience Sim":
        st.markdown("# 🛡️ Resilience Simulation Results")
        st.caption("100 Monte-Carlo disruption runs  ·  MTTF 5yr  ·  MTTR 45d")

        fig_gauges = build_simulation_gauges(sim)
        st.plotly_chart(fig_gauges, key="sim_gauges", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Simulation Parameters</div>', unsafe_allow_html=True)
            param_data = {
                'Parameter': ['MTTF', 'MTTR', 'Simulation Duration', 'Monte Carlo Runs', 'Rerouting Efficiency'],
                'Value': ['1,825 days (5 yr)', '45 days', '3,650 days (10 yr)', '100', '50%'],
            }
            st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)

        with c2:
            st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
            metric_data = {
                'Metric': ['Average Fill Rate', 'Worst-Case Fill Rate', 'Avg Disruptions per Run', 'Resilience Score'],
                'Value': [
                    f"{sim['average_fill_rate']:.1%}",
                    f"{sim['min_fill_rate']:.1%}",
                    f"{sim['average_n_disruptions']:.2f}",
                    f"{sim.get('resilience_score', 0):.1%}",
                ],
            }
            st.dataframe(pd.DataFrame(metric_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
