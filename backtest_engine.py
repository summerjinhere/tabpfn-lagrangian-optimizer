"""
Integrated Backtest Engine for Unified Lagrangian Portfolio
===========================================================

Combines:
1. Unified Lagrangian Optimizer
2. Rolling window backtesting
3. Performance attribution
4. Statistical validation

Author: AI Finance Research
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from unified_lagrangian_optimizer import (
    UnifiedLagrangianOptimizer,
    LagrangianConfig,
    MarketRegime,
    UnifiedPortfolioOptimizationPipeline
)


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Backtest configuration parameters"""
    # Time settings
    train_window: int = 252 * 3  # 3 years training
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    
    # Transaction costs
    transaction_cost_bps: float = 8.0  # NASDAQ calibrated
    
    # Benchmark
    benchmark: str = "equal_weight"  # equal_weight, market_cap, spy
    
    # Statistical validation
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 22
    confidence_level: float = 0.95
    
    # Factor model
    use_ff5: bool = True
    use_momentum: bool = True


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class UnifiedBacktestEngine:
    """
    Rolling Window Backtest Engine for Unified Lagrangian Optimizer
    
    Process:
    1. At each rebalance date:
        a. Estimate expected returns (TabPFN or fallback)
        b. Detect market regime
        c. Run Lagrangian optimization
        d. Execute trades with transaction costs
    2. Track daily returns
    3. Compute performance metrics
    4. Run statistical validation
    """
    
    def __init__(
        self,
        optimizer_config: LagrangianConfig = None,
        backtest_config: BacktestConfig = None
    ):
        self.optimizer_config = optimizer_config or LagrangianConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        
        # Results storage
        self.portfolio_returns = None
        self.benchmark_returns = None
        self.weights_history = None
        self.regime_history = None
        self.turnover_history = None
        self.transaction_costs = None
        
        # Performance metrics
        self.metrics = {}
        self.attribution = {}
    
    def run(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame = None,
        sentiment: pd.DataFrame = None,
        betas: pd.DataFrame = None,
        volatility: pd.DataFrame = None,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """
        Run the backtest.
        
        Parameters:
            returns: Asset returns (T × N)
            factor_returns: Fama-French factors (T × F)
            sentiment: FinBERT sentiment (T × N)
            betas: Kalman betas (T × N)
            volatility: GARCH-MIDAS vol (T × N)
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Dict with performance metrics
        """
        print("=" * 70)
        print("UNIFIED LAGRANGIAN BACKTEST")
        print("=" * 70)
        
        # Filter date range
        if start_date:
            returns = returns[returns.index >= start_date]
        if end_date:
            returns = returns[returns.index <= end_date]
        
        print(f"\nBacktest period: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"Assets: {len(returns.columns)}, Days: {len(returns)}")
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(returns.index)
        print(f"Rebalance frequency: {self.backtest_config.rebalance_frequency}")
        print(f"Rebalance dates: {len(rebalance_dates)}")
        
        # Initialize
        train_window = self.backtest_config.train_window
        n_assets = len(returns.columns)
        
        # Storage
        portfolio_values = [1.0]
        benchmark_values = [1.0]
        weights_hist = {}
        regime_hist = {}
        turnover_hist = {}
        costs_hist = {}
        
        # Current weights
        current_weights = pd.Series(0.0, index=returns.columns)
        
        # Main loop
        print("\n" + "-" * 70)
        print("Running backtest...")
        
        for i, rebal_date in enumerate(rebalance_dates):
            # Get position in index
            idx = returns.index.get_loc(rebal_date)
            
            # Check if enough history
            if idx < train_window:
                continue
            
            # Get training data
            train_returns = returns.iloc[idx - train_window:idx]
            
            # Get auxiliary data if available
            train_sentiment = None
            train_betas = None
            train_vol = None
            
            if sentiment is not None:
                try:
                    train_sentiment = sentiment.iloc[idx - train_window:idx]
                except:
                    pass
            
            if betas is not None:
                try:
                    train_betas = betas.iloc[idx - train_window:idx]
                except:
                    pass
            
            if volatility is not None:
                try:
                    train_vol = volatility.iloc[idx - train_window:idx]
                except:
                    pass
            
            # Filter factor returns
            train_factors = None
            if factor_returns is not None:
                try:
                    train_factors = factor_returns.iloc[idx - train_window:idx]
                except:
                    pass
            
            # Create and fit pipeline
            pipeline = UnifiedPortfolioOptimizationPipeline(self.optimizer_config)
            
            try:
                pipeline.fit(
                    returns=train_returns,
                    factor_returns=train_factors,
                    sentiment=train_sentiment,
                    betas=train_betas,
                    volatility=train_vol
                )
                
                # Get optimal weights
                new_weights = pipeline.optimize()
                regime = pipeline.get_regime()
                
            except Exception as e:
                print(f"⚠️ Optimization failed at {rebal_date}: {e}")
                new_weights = pd.Series(1.0 / n_assets, index=returns.columns)
                regime = MarketRegime.LOW_VOL_BULL
            
            # Align weights
            new_weights = new_weights.reindex(returns.columns).fillna(0)
            
            # Calculate turnover
            turnover = (new_weights - current_weights).abs().sum() / 2
            
            # Transaction costs
            tc_bps = self.backtest_config.transaction_cost_bps
            tc_cost = turnover * tc_bps / 10000
            
            # Store
            weights_hist[rebal_date] = new_weights.copy()
            regime_hist[rebal_date] = regime
            turnover_hist[rebal_date] = turnover
            costs_hist[rebal_date] = tc_cost
            
            # Update weights
            current_weights = new_weights.copy()
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(rebalance_dates)} rebalances")
        
        # Calculate daily returns
        print("\nCalculating daily returns...")
        
        # Create weight matrix
        weight_dates = sorted(weights_hist.keys())
        
        if len(weight_dates) == 0:
            print("❌ No valid rebalance dates!")
            return {}
        
        # Initialize return series
        start_idx = returns.index.get_loc(weight_dates[0])
        daily_returns = []
        daily_dates = []
        benchmark_daily = []
        
        current_weights = weights_hist[weight_dates[0]]
        weight_idx = 0
        
        for i in range(start_idx, len(returns)):
            date = returns.index[i]
            
            # Check if rebalance
            if weight_idx + 1 < len(weight_dates) and date >= weight_dates[weight_idx + 1]:
                weight_idx += 1
                current_weights = weights_hist[weight_dates[weight_idx]]
            
            # Calculate return
            day_returns = returns.iloc[i]
            port_ret = (current_weights * day_returns).sum()
            
            # Apply transaction costs on rebalance dates
            if date in costs_hist:
                port_ret -= costs_hist[date]
            
            # Benchmark (equal weight)
            bench_ret = day_returns.mean()
            
            daily_returns.append(port_ret)
            benchmark_daily.append(bench_ret)
            daily_dates.append(date)
        
        # Store results
        self.portfolio_returns = pd.Series(daily_returns, index=daily_dates)
        self.benchmark_returns = pd.Series(benchmark_daily, index=daily_dates)
        self.weights_history = pd.DataFrame(weights_hist).T
        self.regime_history = pd.Series(regime_hist)
        self.turnover_history = pd.Series(turnover_hist)
        self.transaction_costs = pd.Series(costs_hist)
        
        # Calculate metrics
        print("\nCalculating performance metrics...")
        self.metrics = self._calculate_metrics()
        
        # Factor attribution
        if factor_returns is not None:
            print("Running factor attribution...")
            self.attribution = self._run_attribution(factor_returns)
        
        # Print summary
        self._print_summary()
        
        return self.metrics
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List:
        """Get rebalance dates based on frequency"""
        freq = self.backtest_config.rebalance_frequency
        
        if freq == "daily":
            return list(dates)
        elif freq == "weekly":
            # Every Friday
            return [d for d in dates if d.dayofweek == 4]
        elif freq == "monthly":
            # Last trading day of month
            month_ends = []
            current_month = None
            for d in dates:
                if current_month is not None and d.month != current_month:
                    month_ends.append(prev_date)
                prev_date = d
                current_month = d.month
            return month_ends
        else:
            raise ValueError(f"Unknown frequency: {freq}")
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if self.portfolio_returns is None:
            return {}
        
        port_ret = self.portfolio_returns
        bench_ret = self.benchmark_returns
        
        # Basic statistics
        ann_factor = 252
        
        total_return = (1 + port_ret).prod() - 1
        bench_total_return = (1 + bench_ret).prod() - 1
        
        ann_return = (1 + total_return) ** (ann_factor / len(port_ret)) - 1
        ann_vol = port_ret.std() * np.sqrt(ann_factor)
        
        bench_ann_return = (1 + bench_total_return) ** (ann_factor / len(bench_ret)) - 1
        bench_ann_vol = bench_ret.std() * np.sqrt(ann_factor)
        
        # Risk-adjusted
        rf = self.optimizer_config.risk_free_rate if hasattr(self.optimizer_config, 'risk_free_rate') else 0.03
        sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + port_ret).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (port_ret > 0).mean()
        
        # Information ratio
        excess = port_ret - bench_ret
        ir = excess.mean() / excess.std() * np.sqrt(ann_factor) if excess.std() > 0 else 0
        
        # Turnover
        avg_turnover = self.turnover_history.mean() if self.turnover_history is not None else 0
        
        # Transaction costs
        total_tc = self.transaction_costs.sum() if self.transaction_costs is not None else 0
        
        metrics = {
            # Returns
            'total_return': total_return,
            'annualized_return': ann_return,
            'benchmark_return': bench_total_return,
            'benchmark_ann_return': bench_ann_return,
            'excess_return': ann_return - bench_ann_return,
            
            # Risk
            'annualized_volatility': ann_vol,
            'benchmark_volatility': bench_ann_vol,
            'max_drawdown': max_drawdown,
            
            # Risk-adjusted
            'sharpe_ratio': sharpe,
            'calmar_ratio': calmar,
            'information_ratio': ir,
            
            # Other
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'total_transaction_costs': total_tc,
            'n_days': len(port_ret),
            'n_rebalances': len(self.weights_history) if self.weights_history is not None else 0,
        }
        
        return metrics
    
    def _run_attribution(self, factor_returns: pd.DataFrame) -> Dict:
        """Run Fama-French factor attribution"""
        if not STATSMODELS_AVAILABLE:
            return {}
        
        # Align returns
        excess_returns = self.portfolio_returns - self.benchmark_returns
        
        common_idx = excess_returns.index.intersection(factor_returns.index)
        if len(common_idx) < 30:
            return {'error': 'Insufficient data for attribution'}
        
        y = excess_returns.loc[common_idx]
        X = factor_returns.loc[common_idx]
        
        # Add constant
        X = sm.add_constant(X)
        
        try:
            model = OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            
            attribution = {
                'alpha': model.params.get('const', 0),
                'alpha_tstat': model.tvalues.get('const', 0),
                'alpha_pvalue': model.pvalues.get('const', 1),
                'r_squared': model.rsquared,
                'factor_loadings': model.params.drop('const', errors='ignore').to_dict(),
                'factor_tstats': model.tvalues.drop('const', errors='ignore').to_dict(),
            }
            
        except Exception as e:
            attribution = {'error': str(e)}
        
        return attribution
    
    def _print_summary(self):
        """Print backtest summary"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        
        m = self.metrics
        
        print(f"\n{'RETURNS':^40}")
        print("-" * 40)
        print(f"  Total Return:        {m.get('total_return', 0):.2%}")
        print(f"  Annualized Return:   {m.get('annualized_return', 0):.2%}")
        print(f"  Benchmark Return:    {m.get('benchmark_ann_return', 0):.2%}")
        print(f"  Excess Return:       {m.get('excess_return', 0):.2%}")
        
        print(f"\n{'RISK':^40}")
        print("-" * 40)
        print(f"  Annualized Vol:      {m.get('annualized_volatility', 0):.2%}")
        print(f"  Max Drawdown:        {m.get('max_drawdown', 0):.2%}")
        
        print(f"\n{'RISK-ADJUSTED':^40}")
        print("-" * 40)
        print(f"  Sharpe Ratio:        {m.get('sharpe_ratio', 0):.3f}")
        print(f"  Calmar Ratio:        {m.get('calmar_ratio', 0):.3f}")
        print(f"  Information Ratio:   {m.get('information_ratio', 0):.3f}")
        
        print(f"\n{'TRADING':^40}")
        print("-" * 40)
        print(f"  Win Rate:            {m.get('win_rate', 0):.2%}")
        print(f"  Avg Turnover:        {m.get('avg_turnover', 0):.2%}")
        print(f"  Total TC:            {m.get('total_transaction_costs', 0):.4%}")
        print(f"  N Rebalances:        {m.get('n_rebalances', 0)}")
        
        # Attribution
        if self.attribution and 'alpha' in self.attribution:
            print(f"\n{'FACTOR ATTRIBUTION':^40}")
            print("-" * 40)
            print(f"  Alpha:               {self.attribution.get('alpha', 0):.4%}")
            print(f"  Alpha t-stat:        {self.attribution.get('alpha_tstat', 0):.3f}")
            print(f"  Alpha p-value:       {self.attribution.get('alpha_pvalue', 1):.4f}")
            print(f"  R-squared:           {self.attribution.get('r_squared', 0):.3f}")
            
            if 'factor_loadings' in self.attribution:
                print("\n  Factor Loadings:")
                for f, l in self.attribution['factor_loadings'].items():
                    t = self.attribution.get('factor_tstats', {}).get(f, 0)
                    print(f"    {f:12s}: {l:+.4f} (t={t:.2f})")
        
        print("\n" + "=" * 70)
    
    def get_results(self) -> Dict:
        """Return all results"""
        return {
            'metrics': self.metrics,
            'attribution': self.attribution,
            'portfolio_returns': self.portfolio_returns,
            'benchmark_returns': self.benchmark_returns,
            'weights_history': self.weights_history,
            'regime_history': self.regime_history,
            'turnover_history': self.turnover_history,
        }


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

class StatisticalValidator:
    """
    Statistical Validation for Portfolio Performance
    
    Methods:
    1. Bootstrap confidence intervals
    2. Newey-West HAC standard errors
    3. Multiple testing correction
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def bootstrap_sharpe(
        self,
        returns: pd.Series,
        n_samples: int = None,
        block_size: int = None
    ) -> Dict:
        """
        Bootstrap confidence interval for Sharpe ratio.
        
        Uses block bootstrap to preserve autocorrelation.
        """
        if n_samples is None:
            n_samples = self.config.bootstrap_samples
        if block_size is None:
            block_size = self.config.bootstrap_block_size
        
        n = len(returns)
        n_blocks = n // block_size + 1
        
        sharpes = []
        
        for _ in range(n_samples):
            # Block bootstrap
            blocks = np.random.randint(0, n - block_size, n_blocks)
            boot_idx = []
            for b in blocks:
                boot_idx.extend(range(b, min(b + block_size, n)))
            boot_idx = boot_idx[:n]
            
            boot_returns = returns.iloc[boot_idx]
            sharpe = boot_returns.mean() / boot_returns.std() * np.sqrt(252)
            sharpes.append(sharpe)
        
        sharpes = np.array(sharpes)
        alpha = 1 - self.config.confidence_level
        
        return {
            'mean': np.mean(sharpes),
            'std': np.std(sharpes),
            'lower': np.percentile(sharpes, alpha / 2 * 100),
            'upper': np.percentile(sharpes, (1 - alpha / 2) * 100),
            'p_value_zero': (sharpes <= 0).mean(),
        }
    
    def newey_west_alpha(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame = None,
        maxlags: int = None
    ) -> Dict:
        """
        Test alpha significance with Newey-West HAC standard errors.
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available'}
        
        if maxlags is None:
            maxlags = int(4 * (len(returns) / 100) ** (2 / 9))
        
        if factor_returns is None:
            # Market model
            X = sm.add_constant(np.ones(len(returns)))
            y = returns.values
        else:
            common_idx = returns.index.intersection(factor_returns.index)
            y = returns.loc[common_idx].values
            X = sm.add_constant(factor_returns.loc[common_idx].values)
        
        try:
            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
            
            return {
                'alpha': model.params[0],
                'alpha_se': model.bse[0],
                'alpha_tstat': model.tvalues[0],
                'alpha_pvalue': model.pvalues[0],
            }
        except Exception as e:
            return {'error': str(e)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Unified Lagrangian Backtest Engine - Demo")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_assets = 30
    n_days = 252 * 5  # 5 years
    
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
    
    # Simulate returns
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.02,
        index=dates,
        columns=assets
    )
    
    # Add sector structure
    market_factor = np.random.randn(n_days) * 0.01
    for col in returns.columns:
        returns[col] += market_factor
    
    print(f"\nSample data: {n_days} days × {n_assets} assets")
    
    # Configure
    opt_config = LagrangianConfig(
        lambda_mu=1.0,
        lambda_topology=0.1,
        lambda_ontology=0.2,
        lambda_hedge=0.5,
        long_only=True
    )
    
    bt_config = BacktestConfig(
        train_window=252,
        rebalance_frequency="monthly",
        transaction_cost_bps=8.0
    )
    
    # Run backtest
    engine = UnifiedBacktestEngine(opt_config, bt_config)
    results = engine.run(returns)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
