"""
Feature Engineering for Unified Lagrangian Portfolio
====================================================

Provides feature extraction layers:
1. Kalman Filter Beta (Dynamic Factor Loading)
2. GARCH-MIDAS Volatility
3. FinBERT Sentiment Integration
4. Momentum & Technical Features

Author: AI Finance Research
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# KALMAN FILTER BETA
# =============================================================================

@dataclass
class KalmanConfig:
    """Kalman Filter configuration"""
    observation_noise: float = 0.01
    process_noise: float = 0.001
    initial_beta: float = 1.0
    initial_variance: float = 0.1


class KalmanFilterBeta:
    """
    Kalman Filter for Dynamic Beta Estimation
    
    State-space model:
        β_t = β_{t-1} + η_t,  η_t ~ N(0, Q)  [State equation]
        r_t = α + β_t * r_m,t + ε_t,  ε_t ~ N(0, R)  [Observation equation]
    """
    
    def __init__(self, config: KalmanConfig = None):
        self.config = config or KalmanConfig()
        self.beta = None
        self.variance = None
        self.beta_history = []
        self.variance_history = []
    
    def fit(self, asset_returns: pd.Series, market_returns: pd.Series) -> pd.Series:
        """Run Kalman filter to estimate time-varying beta."""
        common_idx = asset_returns.index.intersection(market_returns.index)
        y = asset_returns.loc[common_idx].values
        x = market_returns.loc[common_idx].values
        
        n = len(y)
        
        beta = self.config.initial_beta
        variance = self.config.initial_variance
        R = self.config.observation_noise
        Q = self.config.process_noise
        
        betas = []
        variances = []
        
        for t in range(n):
            beta_pred = beta
            variance_pred = variance + Q
            
            if abs(x[t]) > 1e-10:
                K = variance_pred * x[t] / (x[t]**2 * variance_pred + R)
                innovation = y[t] - beta_pred * x[t]
                beta = beta_pred + K * innovation
                variance = (1 - K * x[t]) * variance_pred
            else:
                beta = beta_pred
                variance = variance_pred
            
            betas.append(beta)
            variances.append(variance)
        
        self.beta_history = betas
        self.variance_history = variances
        
        return pd.Series(betas, index=common_idx, name='kalman_beta')
    
    def get_current_beta(self) -> float:
        if self.beta_history:
            return self.beta_history[-1]
        return self.config.initial_beta


class KalmanBetaEstimator:
    """Multi-asset Kalman Beta Estimator"""
    
    def __init__(self, config: KalmanConfig = None):
        self.config = config or KalmanConfig()
        self.filters = {}
        self.betas = None
    
    def fit(self, returns: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
        """Estimate dynamic betas for all assets."""
        betas = {}
        
        for col in returns.columns:
            kf = KalmanFilterBeta(self.config)
            betas[col] = kf.fit(returns[col], market_returns)
            self.filters[col] = kf
        
        self.betas = pd.DataFrame(betas)
        return self.betas


# =============================================================================
# GARCH-MIDAS VOLATILITY
# =============================================================================

@dataclass
class GARCHConfig:
    """GARCH-MIDAS configuration"""
    omega: float = 0.00001
    alpha: float = 0.1
    beta: float = 0.85
    midas_k: int = 22


class GARCHMIDASVolatility:
    """GARCH-MIDAS Volatility Model"""
    
    def __init__(self, config: GARCHConfig = None):
        self.config = config or GARCHConfig()
        self.volatility = None
        self.short_term = None
        self.long_term = None
    
    def fit(self, returns: pd.Series) -> pd.Series:
        """Estimate GARCH-MIDAS volatility."""
        n = len(returns)
        r = returns.values
        
        omega = self.config.omega
        alpha = self.config.alpha
        beta = self.config.beta
        k = self.config.midas_k
        
        var_unconditional = np.var(r)
        
        g = np.zeros(n)
        g[0] = 1.0
        
        for t in range(1, n):
            innovation = (r[t-1] / np.sqrt(var_unconditional + 1e-10))**2
            g[t] = (1 - alpha - beta) + alpha * innovation + beta * g[t-1]
        
        rv = pd.Series(r**2).rolling(k).mean().fillna(var_unconditional).values
        
        weights = np.array([(1 - j/k)**2 for j in range(k)])
        weights = weights / weights.sum()
        
        tau = np.zeros(n)
        for t in range(n):
            start = max(0, t - k + 1)
            end = t + 1
            if end - start > 0:
                rv_slice = rv[start:end]
                w_len = min(len(rv_slice), k)
                w_slice = weights[:w_len]
                w_slice = w_slice / (w_slice.sum() + 1e-10)
                # Ensure same length
                rv_slice = rv_slice[-w_len:]
                tau[t] = np.sum(w_slice * rv_slice)
            else:
                tau[t] = var_unconditional
        
        sigma2 = tau * g
        sigma = np.sqrt(np.maximum(sigma2, 1e-10))
        
        self.volatility = pd.Series(sigma, index=returns.index, name='garch_midas_vol')
        self.short_term = pd.Series(g, index=returns.index, name='garch_g')
        self.long_term = pd.Series(tau, index=returns.index, name='midas_tau')
        
        return self.volatility


class GARCHVolatilityEstimator:
    """Multi-asset GARCH-MIDAS Volatility Estimator"""
    
    def __init__(self, config: GARCHConfig = None):
        self.config = config or GARCHConfig()
        self.models = {}
        self.volatility = None
    
    def fit(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate GARCH-MIDAS volatility for all assets."""
        vols = {}
        
        for col in returns.columns:
            model = GARCHMIDASVolatility(self.config)
            vols[col] = model.fit(returns[col])
            self.models[col] = model
        
        self.volatility = pd.DataFrame(vols)
        return self.volatility


# =============================================================================
# SENTIMENT FEATURES
# =============================================================================

@dataclass
class SentimentConfig:
    """Sentiment processing configuration"""
    decay: float = 0.9
    smoothing_window: int = 5
    threshold: float = 0.1


class SentimentProcessor:
    """Sentiment Signal Processing"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
    
    def process(self, sentiment: pd.DataFrame) -> pd.DataFrame:
        """Process raw sentiment scores."""
        sentiment = sentiment.fillna(0)
        smoothed = sentiment.ewm(alpha=1-self.config.decay).mean()
        
        mean_cs = smoothed.mean(axis=1)
        std_cs = smoothed.std(axis=1)
        
        normalized = smoothed.sub(mean_cs, axis=0).div(std_cs + 1e-10, axis=0)
        
        return normalized
    
    def extract_signals(self, sentiment: pd.DataFrame) -> pd.DataFrame:
        """Extract trading signals from sentiment."""
        processed = self.process(sentiment)
        signals = processed.clip(-3, 3) / 3
        return signals


# =============================================================================
# MOMENTUM & TECHNICAL FEATURES
# =============================================================================

class MomentumFeatures:
    """Momentum and Technical Feature Extraction"""
    
    @staticmethod
    def compute_momentum(returns: pd.DataFrame, windows: List[int] = None) -> Dict[str, pd.DataFrame]:
        if windows is None:
            windows = [5, 20, 60, 252]
        
        features = {}
        for w in windows:
            if len(returns) >= w:
                features[f'momentum_{w}d'] = returns.rolling(w).mean()
        return features
    
    @staticmethod
    def compute_vol_adjusted_momentum(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        mu = returns.rolling(window).mean()
        sigma = returns.rolling(window).std()
        return mu / (sigma + 1e-10)
    
    @staticmethod
    def compute_reversal(returns: pd.DataFrame, short_window: int = 5, long_window: int = 60) -> pd.DataFrame:
        short_ret = returns.rolling(short_window).sum()
        long_ret = returns.rolling(long_window).sum()
        return short_ret - long_ret


# =============================================================================
# INTEGRATED FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """Integrated Feature Extractor for TabPFN"""
    
    def __init__(
        self,
        kalman_config: KalmanConfig = None,
        garch_config: GARCHConfig = None,
        sentiment_config: SentimentConfig = None
    ):
        self.kalman_config = kalman_config or KalmanConfig()
        self.garch_config = garch_config or GARCHConfig()
        self.sentiment_config = sentiment_config or SentimentConfig()
        
        self.kalman_estimator = KalmanBetaEstimator(self.kalman_config)
        self.garch_estimator = GARCHVolatilityEstimator(self.garch_config)
        self.sentiment_processor = SentimentProcessor(self.sentiment_config)
        
        self.features = None
        self.feature_names = []
    
    def extract(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series = None,
        sentiment: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extract all features."""
        feature_dict = {}
        
        # Kalman beta
        if market_returns is not None:
            betas = self.kalman_estimator.fit(returns, market_returns)
            for col in betas.columns:
                feature_dict[f'{col}_beta'] = betas[col]
        
        # GARCH-MIDAS volatility
        vols = self.garch_estimator.fit(returns)
        for col in vols.columns:
            feature_dict[f'{col}_vol'] = vols[col]
        
        # Sentiment
        if sentiment is not None:
            sent_signals = self.sentiment_processor.extract_signals(sentiment)
            for col in sent_signals.columns:
                feature_dict[f'{col}_sentiment'] = sent_signals[col]
        
        # Momentum features
        momentum_dict = MomentumFeatures.compute_momentum(returns)
        for name, df in momentum_dict.items():
            for col in df.columns:
                feature_dict[f'{col}_{name}'] = df[col]
        
        # Vol-adjusted momentum
        vol_adj_mom = MomentumFeatures.compute_vol_adjusted_momentum(returns)
        for col in vol_adj_mom.columns:
            feature_dict[f'{col}_vol_adj_mom'] = vol_adj_mom[col]
        
        # Reversal
        reversal = MomentumFeatures.compute_reversal(returns)
        for col in reversal.columns:
            feature_dict[f'{col}_reversal'] = reversal[col]
        
        self.features = pd.DataFrame(feature_dict)
        self.feature_names = list(feature_dict.keys())
        
        return self.features
    
    def get_cross_sectional_features(self, date: pd.Timestamp = None) -> pd.DataFrame:
        """Get cross-sectional features for a specific date."""
        if self.features is None:
            raise ValueError("Must call extract() first")
        
        if date is None:
            date = self.features.index[-1]
        
        row = self.features.loc[date]
        
        # Parse asset names and feature types
        assets = set()
        feature_types = set()
        
        for col in row.index:
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                assets.add(parts[0])
                feature_types.add(parts[1])
            else:
                # Handle multi-word features
                for sep in ['_momentum_', '_vol_adj_', '_']:
                    if sep in col:
                        idx = col.find(sep)
                        assets.add(col[:idx])
                        feature_types.add(col[idx+1:])
                        break
        
        assets = sorted(list(assets))
        feature_types = sorted(list(feature_types))
        
        # Build matrix
        data = np.zeros((len(assets), len(feature_types)))
        
        for i, asset in enumerate(assets):
            for j, feat in enumerate(feature_types):
                col_name = f'{asset}_{feat}'
                if col_name in row.index:
                    val = row[col_name]
                    data[i, j] = val if not np.isnan(val) else 0
        
        return pd.DataFrame(data, index=assets, columns=feature_types)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Feature Engineering - Demo")
    print("=" * 70)
    
    np.random.seed(42)
    n_assets = 10
    n_days = 252 * 2
    
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.02,
        index=dates,
        columns=assets
    )
    
    market = pd.Series(np.random.randn(n_days) * 0.015, index=dates, name='market')
    
    sentiment = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.3,
        index=dates,
        columns=assets
    )
    
    print(f"\nData: {n_days} days × {n_assets} assets")
    
    extractor = FeatureExtractor()
    features = extractor.extract(returns, market, sentiment)
    
    print(f"\nTotal features extracted: {len(features.columns)}")
    print(f"Feature matrix shape: {features.shape}")
    
    cs_features = extractor.get_cross_sectional_features()
    print(f"\nCross-sectional feature matrix: {cs_features.shape}")
    print(f"\nSample features for last date:")
    print(cs_features.head())
    
    print("\n" + "-" * 70)
    print("Kalman Beta Estimation")
    print("-" * 70)
    
    kalman = KalmanBetaEstimator()
    betas = kalman.fit(returns, market)
    print(f"Beta estimates shape: {betas.shape}")
    print(f"Latest betas:\n{betas.iloc[-1]}")
    
    print("\n" + "-" * 70)
    print("GARCH-MIDAS Volatility")
    print("-" * 70)
    
    garch = GARCHVolatilityEstimator()
    vols = garch.fit(returns)
    print(f"Volatility estimates shape: {vols.shape}")
    print(f"Latest volatility:\n{vols.iloc[-1]}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
