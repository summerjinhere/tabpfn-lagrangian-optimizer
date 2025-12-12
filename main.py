"""
Unified Lagrangian Portfolio - Main Runner
==========================================

Complete example showing:
1. Data loading
2. Feature extraction
3. Lagrangian optimization
4. Backtesting
5. Performance analysis

Usage:
    python main.py

Author: AI Finance Research
Date: December 2025
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_lagrangian_optimizer import (
    UnifiedLagrangianOptimizer,
    LagrangianConfig,
    MarketRegime,
    UnifiedPortfolioOptimizationPipeline
)
from backtest_engine import UnifiedBacktestEngine, BacktestConfig
from feature_engineering import FeatureExtractor, KalmanBetaEstimator, GARCHVolatilityEstimator
from data_loader import DataManager, TickerUniverse


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_section(text: str):
    """Print section header."""
    print("\n" + "-" * 50)
    print(f" {text}")
    print("-" * 50)


def run_demo():
    """Run demonstration with simulated data."""
    print_header("UNIFIED LAGRANGIAN PORTFOLIO OPTIMIZER - DEMO")
    
    print("\nThis demo uses simulated data to demonstrate the framework.")
    print("For real data, use the run_backtest() function with actual tickers.")
    
    # Generate simulated data
    print_section("1. Generating Simulated Data")
    
    np.random.seed(42)
    n_assets = 20
    n_days = 252 * 5  # 5 years
    
    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
    
    # Simulate returns with factor structure
    market_factor = np.random.randn(n_days) * 0.01
    sector_factors = [np.random.randn(n_days) * 0.005 for _ in range(4)]
    
    returns_data = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        sector = i // 5
        returns_data[:, i] = (
            market_factor + 
            sector_factors[sector] + 
            np.random.randn(n_days) * 0.015
        )
    
    returns = pd.DataFrame(returns_data, index=dates, columns=tickers)
    
    # Simulated market returns
    market_returns = pd.Series(market_factor, index=dates, name='market')
    
    # Simulated factors (FF5 + Mom)
    factors = pd.DataFrame({
        'Mkt-RF': market_factor,
        'SMB': np.random.randn(n_days) * 0.005,
        'HML': np.random.randn(n_days) * 0.005,
        'RMW': np.random.randn(n_days) * 0.003,
        'CMA': np.random.randn(n_days) * 0.003,
        'Mom': np.random.randn(n_days) * 0.008,
        'RF': np.ones(n_days) * 0.0001,
    }, index=dates)
    
    print(f"  Assets: {n_assets}")
    print(f"  Days: {n_days}")
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    
    # Feature extraction
    print_section("2. Feature Extraction")
    
    extractor = FeatureExtractor()
    features = extractor.extract(returns, market_returns)
    
    print(f"  Total features: {len(features.columns)}")
    print(f"  Feature types: beta, vol, momentum, reversal, vol_adj_mom")
    
    # Kalman beta
    kalman = KalmanBetaEstimator()
    betas = kalman.fit(returns, market_returns)
    print(f"\n  Kalman Beta (last day):")
    for ticker, beta in betas.iloc[-1].items():
        if abs(beta - 1.0) > 0.1:
            print(f"    {ticker}: {beta:.3f}")
    
    # Single period optimization
    print_section("3. Single Period Optimization")
    
    config = LagrangianConfig(
        lambda_mu=1.0,
        lambda_topology=0.1,
        lambda_ontology=0.2,
        lambda_hedge=0.5,
        long_only=True,
        max_position=0.15
    )
    
    optimizer = UnifiedLagrangianOptimizer(config)
    
    # Use last 756 days for estimation
    train_returns = returns.iloc[-756:]
    train_factors = factors.iloc[-756:]
    
    # Simple expected returns (momentum-based)
    expected_returns = train_returns.iloc[-60:].mean()
    
    optimizer.fit(
        returns=train_returns,
        expected_returns=expected_returns,
        factor_returns=train_factors,
        regime=MarketRegime.LOW_VOL_BULL
    )
    
    weights = optimizer.optimize()
    
    print("\n  Optimal Weights (top 10):")
    sorted_weights = weights.abs().sort_values(ascending=False)
    for ticker in sorted_weights.head(10).index:
        print(f"    {ticker}: {weights[ticker]:+.4f}")
    
    # Portfolio decomposition
    decomp = optimizer.get_decomposition()
    
    print("\n  Portfolio Metrics:")
    print(f"    Sum of weights:   {weights.sum():.6f}")
    print(f"    Gross leverage:   {decomp['gross_leverage']:.4f}")
    print(f"    Long exposure:    {decomp['long_exposure']:.4f}")
    print(f"    Num positions:    {decomp['num_positions']}")
    print(f"    Herfindahl:       {decomp['herfindahl']:.4f}")
    
    # Full backtest
    print_section("4. Full Backtest")
    
    bt_config = BacktestConfig(
        train_window=252,
        rebalance_frequency="monthly",
        transaction_cost_bps=8.0
    )
    
    engine = UnifiedBacktestEngine(config, bt_config)
    
    results = engine.run(
        returns=returns,
        factor_returns=factors
    )
    
    # Summary
    print_section("5. Performance Summary")
    
    if results:
        print(f"\n  Returns:")
        print(f"    Total Return:      {results.get('total_return', 0):.2%}")
        print(f"    Annualized Return: {results.get('annualized_return', 0):.2%}")
        print(f"    Excess Return:     {results.get('excess_return', 0):.2%}")
        
        print(f"\n  Risk:")
        print(f"    Volatility:        {results.get('annualized_volatility', 0):.2%}")
        print(f"    Max Drawdown:      {results.get('max_drawdown', 0):.2%}")
        
        print(f"\n  Risk-Adjusted:")
        print(f"    Sharpe Ratio:      {results.get('sharpe_ratio', 0):.3f}")
        print(f"    Information Ratio: {results.get('information_ratio', 0):.3f}")
    
    print_header("DEMO COMPLETE")


def run_backtest(
    tickers: list = None,
    universe: str = "diversified",
    start_date: str = "2010-01-01",
    end_date: str = None
):
    """
    Run full backtest with real data.
    
    Parameters:
        tickers: List of ticker symbols
        universe: Pre-defined universe name
        start_date: Backtest start date
        end_date: Backtest end date
    """
    print_header("UNIFIED LAGRANGIAN PORTFOLIO BACKTEST")
    
    # Load data
    print_section("Loading Data")
    
    dm = DataManager()
    data = dm.load_data(
        tickers=tickers,
        universe=universe,
        start_date=start_date,
        end_date=end_date
    )
    
    returns = data['returns']
    factors = data['factors']
    market = data['market']
    
    # Configure optimizer
    print_section("Configuring Optimizer")
    
    config = LagrangianConfig(
        lambda_mu=1.0,
        lambda_topology=0.1,
        lambda_ontology=0.2,
        lambda_hedge=0.5,
        long_only=True,
        max_position=0.15,
        regime_adaptive=True
    )
    
    print(f"  λ_μ (Alpha):    {config.lambda_mu}")
    print(f"  λ_T (Topology): {config.lambda_topology}")
    print(f"  λ_O (Ontology): {config.lambda_ontology}")
    print(f"  λ_H (Hedge):    {config.lambda_hedge}")
    
    bt_config = BacktestConfig(
        train_window=252 * 3,  # 3 years
        rebalance_frequency="monthly",
        transaction_cost_bps=8.0
    )
    
    print(f"\n  Train window: {bt_config.train_window} days")
    print(f"  Rebalance:    {bt_config.rebalance_frequency}")
    print(f"  TC:           {bt_config.transaction_cost_bps} bps")
    
    # Run backtest
    print_section("Running Backtest")
    
    engine = UnifiedBacktestEngine(config, bt_config)
    results = engine.run(returns, factors)
    
    # Get full results
    full_results = engine.get_results()
    
    return full_results


def compare_strategies():
    """Compare different λ configurations."""
    print_header("STRATEGY COMPARISON")
    
    # Generate data
    np.random.seed(42)
    n_assets = 20
    n_days = 252 * 3
    
    tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    
    market_factor = np.random.randn(n_days) * 0.01
    returns_data = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        returns_data[:, i] = market_factor + np.random.randn(n_days) * 0.02
    
    returns = pd.DataFrame(returns_data, index=dates, columns=tickers)
    
    # Define strategies
    strategies = {
        "Conservative": LagrangianConfig(
            lambda_mu=0.5, lambda_topology=0.3, lambda_hedge=1.0, long_only=True
        ),
        "Balanced": LagrangianConfig(
            lambda_mu=1.0, lambda_topology=0.1, lambda_hedge=0.5, long_only=True
        ),
        "Aggressive": LagrangianConfig(
            lambda_mu=2.0, lambda_topology=0.05, lambda_hedge=0.3, long_only=False
        ),
    }
    
    bt_config = BacktestConfig(train_window=252, rebalance_frequency="monthly")
    
    results = {}
    
    for name, config in strategies.items():
        print(f"\n  Running {name}...")
        engine = UnifiedBacktestEngine(config, bt_config)
        metrics = engine.run(returns)
        results[name] = metrics
    
    # Summary table
    print_section("Results Comparison")
    
    print(f"\n{'Strategy':15s} {'Return':>10s} {'Vol':>10s} {'Sharpe':>10s} {'MaxDD':>10s}")
    print("-" * 55)
    
    for name, metrics in results.items():
        ret = metrics.get('annualized_return', 0) * 100
        vol = metrics.get('annualized_volatility', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        mdd = metrics.get('max_drawdown', 0) * 100
        
        print(f"{name:15s} {ret:>9.2f}% {vol:>9.2f}% {sharpe:>10.3f} {mdd:>9.2f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Lagrangian Portfolio Optimizer")
    parser.add_argument("--mode", choices=["demo", "backtest", "compare"], default="demo",
                        help="Run mode: demo, backtest, or compare")
    parser.add_argument("--tickers", nargs="+", help="List of tickers")
    parser.add_argument("--universe", default="diversified", help="Pre-defined universe")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "backtest":
        run_backtest(
            tickers=args.tickers,
            universe=args.universe,
            start_date=args.start,
            end_date=args.end
        )
    elif args.mode == "compare":
        compare_strategies()
