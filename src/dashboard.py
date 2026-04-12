import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src is in Python path if running from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths (Assuming run from the project root)
DATA_DIR = "./data"
RESULTS_DIR = "./results"

# Configure the Streamlit page
st.set_page_config(
    page_title="MTP-2 Supply Chain Network",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .kpi-card {
        background-color: #f82b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-title { font-size: 1.2rem; font-weight: bold; color: #555; }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #1f77b4; }
    h1, h2, h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load essential supply chain data and experiment results."""
    # Data nodes
    try:
        suppliers = pd.read_csv(f"{DATA_DIR}/suppliers.csv")
        factories = pd.read_csv(f"{DATA_DIR}/factories.csv")
        dcs = pd.read_csv(f"{DATA_DIR}/dcs.csv")
        customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
    except Exception as e:
        st.error(f"Error loading node data: {e}")
        return None, None, None, None, None, None

    # Results
    try:
        pareto = pd.read_csv(f"{RESULTS_DIR}/pareto_front.csv")
        with open(f"{RESULTS_DIR}/experiment_summary.json", 'r') as f:
            summary = json.load(f)
    except Exception as e:
        st.error(f"Error loading results data. Did you run the pipeline? {e}")
        return suppliers, factories, dcs, customers, None, None

    return suppliers, factories, dcs, customers, pareto, summary

@st.cache_data
def load_flows(label):
    """Load flow data for a specific representative solution."""
    formatted_label = label.replace(' ', '_').lower()
    
    sf_flows, fd_flows, dc_flows = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        sf_file = f"{RESULTS_DIR}/flows_supplier_factory_{formatted_label}.csv"
        if os.path.exists(sf_file):
            sf_flows = pd.read_csv(sf_file)
            
        fd_file = f"{RESULTS_DIR}/flows_factory_dc_{formatted_label}.csv"
        if os.path.exists(fd_file):
            fd_flows = pd.read_csv(fd_file)
            
        dc_file = f"{RESULTS_DIR}/flows_dc_customer_{formatted_label}.csv"
        if os.path.exists(dc_file):
            dc_flows = pd.read_csv(dc_file)
            
    except Exception as e:
        st.warning(f"Flow data for '{label}' may be missing or incomplete.")
    
    return sf_flows, fd_flows, dc_flows

def draw_3d_pareto(pareto_df, highlight_sol=None):
    """Draw an interactive 3D Pareto Front."""
    fig = px.scatter_3d(
        pareto_df, 
        x='cost', 
        y='emissions', 
        z='resilience',
        color='resilience',
        color_continuous_scale='Viridis',
        opacity=0.7,
        title='Pareto Front (Cost vs Emissions vs Resilience)'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Total Cost ($)',
            yaxis_title='Total Emissions (kg CO2)',
            zaxis_title='Resilience Score'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def draw_network_map(suppliers, factories, dcs, customers, sf_flows, fd_flows, dc_flows):
    """Draw the supply chain network Map on Indian geography with mode-specific edges."""
    fig = go.Figure()

    # Base Nodes
    fig.add_trace(go.Scattergeo(
        lon=suppliers['longitude'], lat=suppliers['latitude'],
        mode='markers+text', marker=dict(size=14, color='green', symbol='square', line=dict(width=2, color='DarkSlateGrey')),
        text=suppliers['supplier_id'], textposition="top center", name='Suppliers'
    ))
    fig.add_trace(go.Scattergeo(
        lon=factories['longitude'], lat=factories['latitude'],
        mode='markers+text', marker=dict(size=14, color='blue', symbol='triangle-up', line=dict(width=2, color='DarkSlateGrey')),
        text=factories['factory_id'], textposition="top center", name='Factories'
    ))
    fig.add_trace(go.Scattergeo(
        lon=dcs['longitude'], lat=dcs['latitude'],
        mode='markers+text', marker=dict(size=12, color='orange', symbol='diamond', line=dict(width=2, color='DarkSlateGrey')),
        text=dcs['dc_id'], textposition="top center", name='DCs'
    ))
    fig.add_trace(go.Scattergeo(
        lon=customers['longitude'], lat=customers['latitude'],
        mode='markers', marker=dict(size=8, color='red', symbol='circle'),
        text=customers['customer_id'], hoverinfo='text', name='Customers'
    ))

    # Helper function to add flow lines
    def add_flow_lines(flow_df, source_df, target_df, source_id_col, target_id_col, color_dict):
        for _, row in flow_df.iterrows():
            try:
                src = source_df[source_df[source_id_col] == row[source_id_col.replace('_id', '')]].iloc[0]
                tgt = target_df[target_df[target_id_col] == row[target_id_col.replace('_id', '')]].iloc[0]
                
                # Extract mode if present
                m = row.get('mode', 'road')
                line_color = color_dict.get(m, 'gray')
                
                fig.add_trace(go.Scattergeo(
                    lon=[src['longitude'], tgt['longitude']],
                    lat=[src['latitude'], tgt['latitude']],
                    mode='lines',
                    line=dict(width=max(1.0, row['flow'] / 1000.0), color=line_color),
                    opacity=0.6,
                    hoverinfo='text',
                    text=f"Mode: {m}<br>Flow: {row['flow']:.1f}",
                    showlegend=False
                ))
            except Exception as e:
                pass

    # Colors for specific modes based on Pyomo model ('road', 'rail')
    mode_colors = {'road': 'rgba(255, 99, 71, 0.8)', 'rail': 'rgba(30, 144, 255, 0.8)'}

    add_flow_lines(sf_flows, suppliers, factories, 'supplier_id', 'factory_id', mode_colors)
    add_flow_lines(fd_flows, factories, dcs, 'factory_id', 'dc_id', mode_colors)
    add_flow_lines(dc_flows, dcs, customers, 'dc_id', 'customer_id', mode_colors)

    fig.update_layout(
        title_text='Multi-Modal Supply Chain Network (India)',
        showlegend=True,
        geo=dict(
            scope='asia',
            projection_type='mercator',
            showland=True, landcolor='rgba(240, 240, 240, 1)',
            showcountries=True, countrycolor='rgba(200, 200, 200, 1)',
            showsubunits=True, subunitcolor='rgba(220, 220, 220, 1)',
            center=dict(lon=82.0, lat=22.0),
            projection_scale=4.5
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def main():
    st.title("🌱 Green Resilient Supply Chain Network - MTP-2 Dashboard")
    st.markdown("Interactive visualization of Multi-Objective Optimization for Automotive Supply Chain in India.")
    
    suppliers, factories, dcs, customers, pareto, summary = load_data()
    
    if summary is None:
        st.stop()
        
    reps = summary.get('representative_solutions', [])
    rep_labels = [r['label'] for r in reps]
    
    # Sidebar
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio("Select View:", ["Dashboard Overview", "Network Explorer", "Pareto Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Solution Selection")
    
    selected_label = st.sidebar.selectbox("Representative Solution", rep_labels)
    selected_solution = next((r for r in reps if r['label'] == selected_label), reps[0])
    
    # KPIs
    st.markdown("### Solution Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Cost</div>
            <div class="kpi-value">${selected_solution['cost']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Emissions</div>
            <div class="kpi-value">{selected_solution['emissions']:,.0f} kg CO2</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Resilience Score</div>
            <div class="kpi-value">{selected_solution['resilience']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    sf_flows, fd_flows, dc_flows = load_flows(selected_label)

    if view_mode == "Dashboard Overview":
        col_map, col_chart = st.columns([3, 2])
        
        with col_map:
            st.subheader(f"Logistics Map - {selected_label}")
            fig_map = draw_network_map(suppliers, factories, dcs, customers, sf_flows, fd_flows, dc_flows)
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col_chart:
            st.subheader("Trade-off Space")
            fig_pareto = draw_3d_pareto(pareto, selected_solution)
            st.plotly_chart(fig_pareto, use_container_width=True)
            
            st.markdown("""
            **Map Legend:**
            - <span style="color:rgba(255, 99, 71, 1); font-weight:bold;">Red Lines</span>: Road Transport (Faster, Higher Emissions)
            - <span style="color:rgba(30, 144, 255, 1); font-weight:bold;">Blue Lines</span>: Rail Transport (Slower, Lower Emissions, Cheaper)
            """, unsafe_allow_html=True)

    elif view_mode == "Network Explorer":
        st.subheader("Interactive Map Explorer")
        fig_map = draw_network_map(suppliers, factories, dcs, customers, sf_flows, fd_flows, dc_flows)
        fig_map.update_layout(height=700)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Display Flow Tables
        st.markdown("### Transport Mode Utilization")
        
        # Aggregate flow by mode
        all_flows = []
        if not sf_flows.empty:
            sf_agg = sf_flows.groupby('mode')['flow'].sum().reset_index()
            sf_agg['stage'] = 'Supplier -> Factory'
            all_flows.append(sf_agg)
        if not fd_flows.empty:
            fd_agg = fd_flows.groupby('mode')['flow'].sum().reset_index()
            fd_agg['stage'] = 'Factory -> DC'
            all_flows.append(fd_agg)
        if not dc_flows.empty:
            dc_agg = dc_flows.groupby('mode')['flow'].sum().reset_index()
            dc_agg['stage'] = 'DC -> Customer'
            all_flows.append(dc_agg)
            
        if all_flows:
            flow_summary = pd.concat(all_flows)
            fig_bar = px.bar(flow_summary, x="stage", y="flow", color="mode", barmode="group",
                            color_discrete_map={'road': 'tomato', 'rail': 'dodgerblue'},
                            title="Volume Transported by Mode per Stage")
            st.plotly_chart(fig_bar, use_container_width=True)

    elif view_mode == "Pareto Analysis":
        st.subheader("Multi-Objective Pareto Trade-offs")
        fig_pareto = draw_3d_pareto(pareto)
        fig_pareto.update_layout(height=800)
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # 2D Projections
        st.markdown("### 2D Projections")
        col1, col2 = st.columns(2)
        with col1:
            fig_2d1 = px.scatter(pareto, x='cost', y='emissions', color='resilience', 
                                title='Cost vs Emissions')
            st.plotly_chart(fig_2d1, use_container_width=True)
        with col2:
            fig_2d2 = px.scatter(pareto, x='cost', y='resilience', color='emissions',
                                title='Cost vs Resilience')
            st.plotly_chart(fig_2d2, use_container_width=True)

if __name__ == "__main__":
    main()
