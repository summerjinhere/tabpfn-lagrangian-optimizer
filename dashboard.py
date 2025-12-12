"""
Unified Lagrangian Portfolio - Dashboard
========================================

Streamlit dashboard for:
1. Interactive Backtesting
2. Regime Analysis
3. Performance Attribution
4. Statistical Validation

Author: AI Finance Research
Date: December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_lagrangian_optimizer import (
    UnifiedLagrangianOptimizer,
    LagrangianConfig,
    MarketRegime
)
from backtest_engine import UnifiedBacktestEngine, BacktestConfig
from data_loader import DataManager, TickerUniverse

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Unified Lagrangian Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UTILS
# =============================================================================

@st.cache_data
def load_data(universe, custom_tickers, start_date):
    """Load and cache data."""
    dm = DataManager()
    
    if universe == "Custom":
        tickers = [t.strip() for t in custom_tickers.split(",")]
    else:
        tickers = None
    
    data = dm.load_data(
        tickers=tickers,
        universe=universe.lower() if universe != "Custom" else None,
        start_date=str(start_date)
    )
    return data

@st.cache_data
def get_benchmark_data(tickers, start_date):
    """Load benchmark data."""
    dm = DataManager()
    data = dm.load_data(tickers=tickers, start_date=str(start_date))
    # Return cumulative returns
    returns = data['returns']
    return (1 + returns).cumprod()

def plot_equity_curve(portfolio, benchmark, regimes=None):
    """Plot interactive equity curve."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Equity Curve
    fig.add_trace(
        go.Scatter(x=portfolio.index, y=portfolio, name="Portfolio",
                  line=dict(color="#2962FF", width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=benchmark.index, y=benchmark, name="Benchmark",
                  line=dict(color="#B0BEC5", width=2, dash='dash')),
        row=1, col=1
    )
    
    # Drawdown
    dd_port = portfolio / portfolio.cummax() - 1
    dd_bench = benchmark / benchmark.cummax() - 1
    
    fig.add_trace(
        go.Scatter(x=dd_port.index, y=dd_port, name="Portfolio DD",
                  fill='tozeroy', line=dict(color="#EF5350", width=1)),
        row=2, col=1
    )
    
    # Regime background
    if regimes is not None:
        # todo: add regime colored bands
        pass
        
    fig.update_layout(
        title="Portfolio Performance",
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    fig.update_yaxes(title_text="Wealth Index", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Selection
    st.subheader("1. Data Selection")
    universe = st.selectbox(
        "Universe",
        ["Diversified", "Tech", "SP500_Top50", "Custom"],
        index=0
    )
    
    custom_tickers = ""
    if universe == "Custom":
        custom_tickers = st.text_area(
            "Enter Tickers (comma separated)",
            "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, JPM, V, JNJ, PFE"
        )
    
    date_col1, date_col2 = st.columns(2)
    start_date = date_col1.date_input("Start Date", datetime(2018, 1, 1))
    end_date = date_col2.date_input("End Date", datetime.now())
    
    # Backtest Settings
    st.subheader("2. Backtest Settings")
    train_window = st.number_input("Train Window (Days)", 126, 1000, 252*2)
    rebalance = st.selectbox("Rebalance", ["Monthly", "Weekly", "Daily"])
    tc_bps = st.number_input("Trans. Cost (bps)", 0.0, 50.0, 8.0)
    
    # Model Parameters
    st.subheader("3. Model Parameters")
    
    with st.expander("Lagrangian Multipliers", expanded=True):
        lambda_mu = st.slider("λ_μ (Alpha)", 0.0, 5.0, 1.0, 0.1)
        lambda_topo = st.slider("λ_T (Topology)", 0.0, 1.0, 0.1, 0.05)
        lambda_onto = st.slider("λ_O (Ontology)", 0.0, 1.0, 0.2, 0.05)
        lambda_hedge = st.slider("λ_H (Hedge)", 0.0, 2.0, 0.5, 0.1)
    
    with st.expander("Constraints"):
        long_only = st.checkbox("Long Only", True)
        max_pos = st.slider("Max Position", 0.05, 1.0, 0.15, 0.05)
        regime_adaptive = st.checkbox("Regime Adaptive", True)

    run_btn = st.button("🚀 Run Backtest", use_container_width=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("Unified Lagrangian Portfolio")

if run_btn:
    with st.spinner("Loading data and running backtest..."):
        # 1. Load Data
        try:
            data = load_data(universe, custom_tickers, start_date)
            
            if len(data['returns']) == 0:
                st.error("No data found! Please check your tickers or date range.")
                st.stop()
                
            # Filter by date
            returns = data['returns']
            mask = (returns.index >= pd.Timestamp(start_date)) & (returns.index <= pd.Timestamp(end_date))
            returns = returns.loc[mask]
            
            if len(returns) < train_window + 20:
                st.error(f"Insufficient data! Need at least {train_window} days for training.")
                st.stop()
                
            factors = data['factors']
            if factors is not None:
                factors = factors.loc[returns.index.intersection(factors.index)]
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
        
        # 2. Configure Engines
        opt_config = LagrangianConfig(
            lambda_mu=lambda_mu,
            lambda_topology=lambda_topo,
            lambda_ontology=lambda_onto,
            lambda_hedge=lambda_hedge,
            long_only=long_only,
            max_position=max_pos,
            regime_adaptive=regime_adaptive
        )
        
        bt_config = BacktestConfig(
            train_window=train_window,
            rebalance_frequency=rebalance.lower(),
            transaction_cost_bps=tc_bps
        )
        
        # 3. Run Backtest
        engine = UnifiedBacktestEngine(opt_config, bt_config)
        
        # Run with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # We need to capture the progress from the engine, 
        # but for now we'll just run it as is since the engine prints to stdout.
        # Ideally we'd modify the engine to accept a callback.
        
        results = engine.run(returns, factors)
        full_results = engine.get_results()
        
        progress_bar.progress(100)
        
        # 4. Display Results
        
        # Extract series
        port_ret = full_results['portfolio_returns']
        bench_ret = full_results['benchmark_returns']
        metrics = full_results['metrics']
        
        # Cumulative
        port_cum = (1 + port_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()
        
        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Summary", "📈 Analysis", "🔄 Regimes", "🔍 Attribution", "📉 Drawdown"
        ])
        
        with tab1:
            # Metrics Row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Return", f"{metrics['total_return']:.1%}")
            c2.metric("Ann. Return", f"{metrics['annualized_return']:.1%}")
            c3.metric("Ann. Volatility", f"{metrics['annualized_volatility']:.1%}")
            c4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            c5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
            
            # Plot
            fig = plot_equity_curve(port_cum, bench_cum)
            st.plotly_chart(fig, use_container_width=True)
        
            # Monthly Heatmap
            st.subheader("Monthly Returns Heatmap")
            
            try:
                # Group by year and month
                monthly_ret = port_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_ret = monthly_ret.to_frame(name='return')
                monthly_ret['year'] = monthly_ret.index.year
                monthly_ret['month'] = monthly_ret.index.month
                
                pivot_table = monthly_ret.pivot(index='year', columns='month', values='return')
                
                # Plot heatmap
                fig_heat = px.imshow(
                    pivot_table,
                    labels=dict(x="Month", y="Year", color="Return"),
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=pivot_table.index,
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0
                )
                fig_heat.update_layout(title="Monthly Returns (%)")
                st.plotly_chart(fig_heat, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")

        with tab2:
            st.header("Portfolio Composition")
            weights = full_results['weights_history']
            if weights is not None:
                fig_w = px.area(weights, title="Asset Allocation")
                st.plotly_chart(fig_w, use_container_width=True)
            
            st.subheader("Turnover")
            turnover = full_results['turnover_history']
            if turnover is not None:
                st.bar_chart(turnover)

        with tab3:
            st.header("Market Regimes")
            regimes = full_results['regime_history']
            if regimes is not None:
                # Convert enum to string for plotting
                regime_str = regimes.apply(lambda x: x.name if hasattr(x, 'name') else str(x))
                fig_r = px.scatter(regime_str, title="Regime History")
                st.plotly_chart(fig_r, use_container_width=True)
                
                # Regime Stats
                st.write(regime_str.value_counts())

        with tab4:
            st.header("Factor Attribution")
            attr = full_results['attribution']
            if attr and 'factor_loadings' in attr:
                loadings = pd.Series(attr['factor_loadings'])
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Alpha (Ann.)", f"{attr['alpha']*252:.2%}")
                    st.metric("R-Squared", f"{attr['r_squared']:.3f}")
                
                with c2:
                    fig_f = px.bar(loadings, orientation='h', title="Factor Exposures")
                    st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.warning("Insufficient data for factor attribution or statsmodels not available.")
                
            # Benchmark Comparison
            st.subheader("Benchmark Comparison")
            bench_tickers = st.multiselect("Select Benchmarks", ["SPY", "QQQ", "IWM", "GLD", "TLT"], ["SPY", "QQQ"])
            if bench_tickers:
                bench_data = get_benchmark_data(bench_tickers, start_date)
                # Align dates
                bench_data = bench_data.loc[port_cum.index.intersection(bench_data.index)]
                
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Portfolio", line=dict(width=3)))
                
                for col in bench_data.columns:
                    # Normalize to start at 1
                    series = bench_data[col]
                    series = series / series.iloc[0]
                    fig_b.add_trace(go.Scatter(x=series.index, y=series, name=col))
                    
                st.plotly_chart(fig_b, use_container_width=True)

        with tab5:
            st.header("Statistical Validation")
            
            from backtest_engine import StatisticalValidator
            validator = StatisticalValidator()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Bootstrap Analysis (Sharpe Ratio)")
                if len(port_ret) > 0:
                    with st.spinner("Running bootstrap analysis..."):
                        boot_res = validator.bootstrap_sharpe(port_ret)
                        
                        st.write(f"**Mean Sharpe:** {boot_res['mean']:.3f}")
                        st.write(f"**95% CI:** [{boot_res['lower']:.3f}, {boot_res['upper']:.3f}]")
                        st.write(f"**Prob(Sharpe <= 0):** {boot_res['p_value_zero']:.4f}")
                        
                        # Generate distribution for plotting
                        # We re-run a small simulation for the plot since we didn't save the array
                        # In production we might want to modify validator to return the array
                        sharpes = []
                        n = len(port_ret)
                        for _ in range(1000):
                            idx = np.random.randint(0, n, n)
                            s = port_ret.iloc[idx].mean() / port_ret.iloc[idx].std() * np.sqrt(252)
                            sharpes.append(s)
                        
                        fig_boot = px.histogram(sharpes, nbins=30, title="Bootstrap Distribution of Sharpe Ratio")
                        fig_boot.add_vline(x=0, line_dash="dash", line_color="red")
                        fig_boot.add_vline(x=metrics['sharpe_ratio'], line_dash="solid", line_color="green", annotation_text="Observed")
                        st.plotly_chart(fig_boot, use_container_width=True)
                else:
                    st.warning("Insufficient data for bootstrap.")

            with col2:
                st.subheader("Alpha Significance (Newey-West)")
                if factors is not None:
                    nw_res = validator.newey_west_alpha(port_ret, factors)
                    
                    if 'error' not in nw_res:
                        st.metric("Alpha (Daily)", f"{nw_res['alpha']:.5f}")
                        st.metric("t-stat", f"{nw_res['alpha_tstat']:.3f}")
                        st.metric("p-value", f"{nw_res['alpha_pvalue']:.4f}")
                        
                        if nw_res['alpha_pvalue'] < 0.05:
                            st.success("Alpha is statistically significant at 5% level!")
                        else:
                            st.info("Alpha is not statistically significant.")
                    else:
                        st.error(f"Error: {nw_res.get('error')}")
                else:
                    st.warning("Factor data required for alpha test.")
                    
            st.markdown("---")
            st.subheader("Rolling Statistics")
            
            window = st.slider("Rolling Window", 21, 252, 63)
            
            roll_vol = port_ret.rolling(window).std() * np.sqrt(252)
            roll_sharpe = port_ret.rolling(window).mean() / port_ret.rolling(window).std() * np.sqrt(252)
            
            fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            fig_roll.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, name="Rolling Vol"), row=1, col=1)
            fig_roll.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name="Rolling Sharpe"), row=2, col=1)
            
            fig_roll.update_layout(height=500, title=f"Rolling {window}-Day Metrics")
            st.plotly_chart(fig_roll, use_container_width=True)

else:
    st.info("👈 Please configure settings in the sidebar and click 'Run Backtest' to start.")
