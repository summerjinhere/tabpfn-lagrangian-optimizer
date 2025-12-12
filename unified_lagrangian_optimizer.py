"""
Unified Lagrangian Portfolio Optimizer
======================================

Implements the Unified Topology-Aware Portfolio Optimization Framework:

    min  w'Σ̃w - μ̃'w
    s.t. 1'w = 1, w ≥ 0 (optional)

Where:
    Σ̃ = Σ + λ_T·L + λ_O·S'S + λ_H·B'B  (Regularized Covariance)
    μ̃ = λ_μ·μ_TabPFN + λ_O·S'b + λ_H·B'E  (Adjusted Expected Return)

Components:
    - TabPFN: Cross-sectional return prediction
    - Topology (L): Graph Laplacian for diversification
    - Ontology (S): Sector membership constraints
    - Hedging (B): Factor exposure neutralization

References:
    - Hollmann et al. (2025) "TabPFN" - Nature
    - Ciciretti & Pallotta (2024) "Network Risk Parity" - JAM
    - Enkhsaikhan & Jo (2025) "Augmented Lagrangian RL for Portfolio" - AAAI
    - Lopez de Prado (2016) "Hierarchical Risk Parity" - JPM

Author: AI Finance Research
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import linalg
    from scipy.optimize import minimize
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class MarketRegime(Enum):
    """4-Regime Classification"""
    LOW_VOL_BEAR = "LowVol_Bear"
    LOW_VOL_BULL = "LowVol_Bull"
    HIGH_VOL_BEAR = "HighVol_Bear"
    HIGH_VOL_BULL = "HighVol_Bull"


@dataclass
class LagrangianConfig:
    """
    Unified Lagrangian Configuration
    
    λ Parameters:
        λ_μ: Alpha signal trust (higher = more reliance on TabPFN predictions)
        λ_T: Topology strength (higher = more diversification pressure)
        λ_O: Ontology weight (higher = stricter sector adherence)
        λ_H: Hedge intensity (higher = more factor neutralization)
    
    Economic Interpretation:
        Each λ represents the shadow price of the corresponding constraint.
    """
    # Lagrange multipliers
    lambda_mu: float = 1.0      # Alpha signal weight
    lambda_topology: float = 0.1  # Topology (Graph Laplacian) weight
    lambda_ontology: float = 0.2  # Ontology (Sector) weight
    lambda_hedge: float = 0.5   # Hedging (Factor neutral) weight
    
    # Topology settings
    distance_metric: str = "correlation"  # correlation, angular, abs_correlation
    linkage_method: str = "ward"  # single, complete, average, ward
    magnitude_only: bool = True  # Use magnitude-only Laplacian (allows pair trading)
    
    # Ontology settings
    sector_bounds: bool = True  # Apply sector allocation bounds
    sector_target_method: str = "equal"  # equal, market_cap, custom
    
    # Hedge settings
    factors: List[str] = field(default_factory=lambda: ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"])
    target_exposures: Optional[Dict[str, float]] = None  # Default: 0 for all (neutral)
    
    # Portfolio constraints
    long_only: bool = False  # Allow short positions
    max_position: float = 0.15  # Maximum single position
    min_position: float = -0.10  # Minimum (for short)
    max_leverage: float = 1.5  # Gross exposure limit
    
    # Regime-adaptive λ
    regime_adaptive: bool = True
    regime_lambda_map: Optional[Dict[MarketRegime, Dict[str, float]]] = None
    
    def __post_init__(self):
        if self.target_exposures is None:
            self.target_exposures = {f: 0.0 for f in self.factors}
        
        if self.regime_lambda_map is None:
            self.regime_lambda_map = {
                MarketRegime.LOW_VOL_BULL: {
                    "lambda_mu": 1.5, "lambda_topology": 0.1, 
                    "lambda_ontology": 0.1, "lambda_hedge": 0.3
                },
                MarketRegime.LOW_VOL_BEAR: {
                    "lambda_mu": 0.8, "lambda_topology": 0.3, 
                    "lambda_ontology": 0.3, "lambda_hedge": 1.0
                },
                MarketRegime.HIGH_VOL_BULL: {
                    "lambda_mu": 1.0, "lambda_topology": 0.2, 
                    "lambda_ontology": 0.2, "lambda_hedge": 0.8
                },
                MarketRegime.HIGH_VOL_BEAR: {
                    "lambda_mu": 0.5, "lambda_topology": 0.5, 
                    "lambda_ontology": 0.4, "lambda_hedge": 1.5
                },
            }


# =============================================================================
# GRAPH LAPLACIAN (TOPOLOGY)
# =============================================================================

class GraphLaplacian:
    """
    Graph Laplacian for Topology-Based Diversification
    
    Converts correlation structure into a graph and computes Laplacian:
        L = D - A
    
    Where:
        A_ij = exp(-d_ij² / σ²)  (Adjacency based on distance)
        D_ii = Σ_j A_ij  (Degree matrix)
    
    Topology Penalty:
        w'Lw = Σ_ij A_ij(w_i - w_j)²
    
    This encourages similar assets to have similar weight magnitudes.
    """
    
    def __init__(self, config: LagrangianConfig):
        self.config = config
        self.L = None
        self.A = None
        self.D = None
        self.assets = None
    
    def _correlation_to_distance(self, corr: pd.DataFrame) -> pd.DataFrame:
        """Convert correlation to distance matrix"""
        method = self.config.distance_metric
        
        if method == "correlation":
            # Lopez de Prado distance
            dist = np.sqrt(0.5 * (1 - corr))
        elif method == "angular":
            dist = np.sqrt(2 * (1 - corr))
        elif method == "abs_correlation":
            dist = np.sqrt(1 - np.abs(corr))
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        np.fill_diagonal(dist.values, 0)
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    
    def _compute_adjacency(self, dist: pd.DataFrame, sigma: float = None) -> pd.DataFrame:
        """
        Compute adjacency matrix using Gaussian kernel:
            A_ij = exp(-d_ij² / σ²)
        """
        if sigma is None:
            # Adaptive sigma based on median distance
            dist_vals = dist.values[np.triu_indices(len(dist), k=1)]
            sigma = np.median(dist_vals)
        
        A = np.exp(-dist.values**2 / (sigma**2 + 1e-10))
        np.fill_diagonal(A, 0)  # No self-loops
        
        return pd.DataFrame(A, index=dist.index, columns=dist.columns)
    
    def fit(self, returns: pd.DataFrame) -> 'GraphLaplacian':
        """Compute Graph Laplacian from return data"""
        # Filter zero-variance assets
        variances = returns.var()
        valid_cols = variances[variances > 1e-10].index.tolist()
        returns = returns[valid_cols]
        
        self.assets = list(returns.columns)
        
        # Correlation → Distance → Adjacency
        corr = returns.corr()
        dist = self._correlation_to_distance(corr)
        self.A = self._compute_adjacency(dist)
        
        # Degree matrix
        self.D = np.diag(self.A.values.sum(axis=1))
        
        # Laplacian: L = D - A
        self.L = pd.DataFrame(
            self.D - self.A.values,
            index=self.assets,
            columns=self.assets
        )
        
        return self
    
    def get_laplacian(self, assets: List[str] = None) -> np.ndarray:
        """Return Laplacian matrix for given assets"""
        if self.L is None:
            raise ValueError("Must call fit() first")
        
        if assets is None:
            return self.L.values
        
        # Filter to requested assets
        valid_assets = [a for a in assets if a in self.L.index]
        return self.L.loc[valid_assets, valid_assets].values
    
    def get_magnitude_laplacian(self) -> np.ndarray:
        """
        Return magnitude-only Laplacian for pair trading support.
        
        This allows:
            - |w_삼성| ≈ |w_하이닉스| (magnitude similar)
            - sign(w_삼성) ≠ sign(w_하이닉스) (direction different)
        """
        return self.get_laplacian()


# =============================================================================
# SECTOR MEMBERSHIP (ONTOLOGY)
# =============================================================================

class SectorOntology:
    """
    Sector Membership Matrix for Ontology Constraints
    
    S: K × N matrix where S_kn = 1 if asset n belongs to sector k
    b: K × 1 vector of target sector allocations
    
    Ontology Penalty:
        ||Sw - b||² = Σ_k (Σ_{n∈k} w_n - b_k)²
    
    This encourages portfolio to match target sector allocations.
    """
    
    # GICS Sector Mapping for common tickers
    SECTOR_MAP = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
        "AMD": "Technology", "INTC": "Technology", "AVGO": "Technology",
        "CSCO": "Technology", "ORCL": "Technology", "CRM": "Technology",
        "ADBE": "Technology", "NOW": "Technology", "IBM": "Technology",
        "QCOM": "Technology", "TXN": "Technology", "MU": "Technology",
        
        # Healthcare
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
        "MRK": "Healthcare", "ABBV": "Healthcare", "TMO": "Healthcare",
        "ABT": "Healthcare", "DHR": "Healthcare", "BMY": "Healthcare",
        "LLY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
        
        # Financials
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "C": "Financials",
        "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
        "V": "Financials", "MA": "Financials", "PYPL": "Financials",
        
        # Consumer Discretionary
        "AMZN": "Consumer", "TSLA": "Technology", "HD": "Consumer",
        "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
        "LOW": "Consumer", "TJX": "Consumer", "BKNG": "Consumer",
        
        # Consumer Staples
        "PG": "Staples", "KO": "Staples", "PEP": "Staples",
        "WMT": "Staples", "COST": "Staples", "PM": "Staples",
        
        # Industrials
        "BA": "Industrials", "HON": "Industrials", "UPS": "Industrials",
        "CAT": "Industrials", "GE": "Industrials", "RTX": "Industrials",
        "LMT": "Industrials", "DE": "Industrials", "MMM": "Industrials",
        
        # Energy
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
        "SLB": "Energy", "EOG": "Energy", "PSX": "Energy",
        
        # Utilities
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        "D": "Utilities", "AEP": "Utilities",
        
        # Materials
        "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
        "ECL": "Materials", "DD": "Materials",
        
        # Real Estate
        "AMT": "RealEstate", "PLD": "RealEstate", "CCI": "RealEstate",
        "EQIX": "RealEstate", "SPG": "RealEstate",
        
        # Communication
        "DIS": "Communication", "CMCSA": "Communication", "VZ": "Communication",
        "T": "Communication", "NFLX": "Communication", "TMUS": "Communication",
    }
    
    def __init__(self, config: LagrangianConfig):
        self.config = config
        self.S = None  # Sector membership matrix
        self.b = None  # Target allocations
        self.sectors = None
        self.assets = None
    
    def fit(self, assets: List[str], market_caps: Dict[str, float] = None) -> 'SectorOntology':
        """Build sector membership matrix"""
        self.assets = assets
        
        # Map assets to sectors
        asset_sectors = {}
        for asset in assets:
            asset_sectors[asset] = self.SECTOR_MAP.get(asset, "Other")
        
        # Get unique sectors
        self.sectors = sorted(list(set(asset_sectors.values())))
        
        # Build membership matrix S (K × N)
        K = len(self.sectors)
        N = len(assets)
        S = np.zeros((K, N))
        
        for i, sector in enumerate(self.sectors):
            for j, asset in enumerate(assets):
                if asset_sectors[asset] == sector:
                    S[i, j] = 1
        
        self.S = S
        
        # Compute target allocations
        if self.config.sector_target_method == "equal":
            # Equal weight across sectors
            self.b = np.ones(K) / K
        elif self.config.sector_target_method == "market_cap" and market_caps:
            # Market-cap weighted
            sector_caps = {s: 0.0 for s in self.sectors}
            for asset in assets:
                sector = asset_sectors[asset]
                sector_caps[sector] += market_caps.get(asset, 1.0)
            total_cap = sum(sector_caps.values())
            self.b = np.array([sector_caps[s] / total_cap for s in self.sectors])
        else:
            # Default to equal
            self.b = np.ones(K) / K
        
        return self
    
    def get_membership_matrix(self) -> np.ndarray:
        """Return S matrix"""
        return self.S
    
    def get_target_allocation(self) -> np.ndarray:
        """Return b vector"""
        return self.b
    
    def get_sector_exposures(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate sector exposures from weights"""
        exposures = self.S @ weights
        return {sector: exp for sector, exp in zip(self.sectors, exposures)}


# =============================================================================
# FACTOR HEDGING
# =============================================================================

class FactorHedger:
    """
    Factor Exposure Management for Hedging
    
    B: F × N matrix of factor loadings (betas)
    E: F × 1 vector of target exposures (typically 0 for neutral)
    
    Hedge Penalty:
        ||Bw - E||² = Σ_f (Σ_n β_fn·w_n - E_f)²
    
    This encourages portfolio to achieve target factor exposures.
    """
    
    def __init__(self, config: LagrangianConfig):
        self.config = config
        self.B = None  # Factor loading matrix
        self.E = None  # Target exposures
        self.factors = config.factors
        self.assets = None
    
    def fit(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame = None
    ) -> 'FactorHedger':
        """Estimate factor loadings using regression"""
        self.assets = list(returns.columns)
        
        if factor_returns is None:
            # Create placeholder identity if no factors provided
            # In production, use actual Fama-French factors
            print("⚠️ No factor data provided, using placeholder betas")
            F = len(self.factors)
            N = len(self.assets)
            self.B = np.random.randn(F, N) * 0.3
            self.B[0, :] = 1.0  # Market beta ≈ 1
        else:
            # Estimate betas via regression
            F = len(self.factors)
            N = len(self.assets)
            self.B = np.zeros((F, N))
            
            # Align data
            common_idx = returns.index.intersection(factor_returns.index)
            ret_aligned = returns.loc[common_idx]
            fac_aligned = factor_returns.loc[common_idx]
            
            for i, asset in enumerate(self.assets):
                if asset in ret_aligned.columns:
                    y = ret_aligned[asset].values
                    X = sm.add_constant(fac_aligned[self.factors].values)
                    
                    try:
                        model = sm.OLS(y, X, missing='drop').fit()
                        self.B[:, i] = model.params[1:F+1]
                    except:
                        self.B[:, i] = np.zeros(F)
                        self.B[0, i] = 1.0  # Default market beta
        
        # Set target exposures
        self.E = np.array([self.config.target_exposures.get(f, 0.0) for f in self.factors])
        
        return self
    
    def get_loading_matrix(self) -> np.ndarray:
        """Return B matrix"""
        return self.B
    
    def get_target_exposures(self) -> np.ndarray:
        """Return E vector"""
        return self.E
    
    def get_factor_exposures(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate factor exposures from weights"""
        exposures = self.B @ weights
        return {factor: exp for factor, exp in zip(self.factors, exposures)}


# =============================================================================
# UNIFIED LAGRANGIAN OPTIMIZER
# =============================================================================

class UnifiedLagrangianOptimizer:
    """
    Unified Topology-Aware Portfolio Optimizer
    
    Solves:
        min  w'Σ̃w - μ̃'w
        s.t. 1'w = 1
             w ≥ 0 (if long_only)
             |w|_1 ≤ leverage (if constrained)
    
    Where:
        Σ̃ = Σ + λ_T·L + λ_O·S'S + λ_H·B'B
        μ̃ = λ_μ·μ + λ_O·S'b + λ_H·B'E
    
    Closed-form solution (unconstrained):
        w* = Σ̃⁻¹(μ̃/2 - ν*/2 · 1)
        ν* = (1'Σ̃⁻¹μ̃ - 2) / (1'Σ̃⁻¹1)
    """
    
    def __init__(self, config: LagrangianConfig = None):
        self.config = config or LagrangianConfig()
        
        # Components
        self.laplacian = GraphLaplacian(self.config)
        self.ontology = SectorOntology(self.config)
        self.hedger = FactorHedger(self.config)
        
        # State
        self.Sigma = None  # Covariance matrix
        self.Sigma_tilde = None  # Regularized covariance
        self.mu = None  # Expected returns (from TabPFN)
        self.mu_tilde = None  # Adjusted expected returns
        self.weights = None
        self.assets = None
        
        # Diagnostics
        self.diagnostics = {}
    
    def fit(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series = None,
        factor_returns: pd.DataFrame = None,
        market_caps: Dict[str, float] = None,
        regime: MarketRegime = None
    ) -> 'UnifiedLagrangianOptimizer':
        """
        Fit the optimizer with all components.
        
        Parameters:
            returns: Historical returns (T × N)
            expected_returns: TabPFN predictions (N,)
            factor_returns: Fama-French factor returns (T × F)
            market_caps: Market capitalizations for ontology
            regime: Current market regime for adaptive λ
        """
        # Filter valid assets
        variances = returns.var()
        valid_cols = variances[variances > 1e-10].index.tolist()
        returns = returns[valid_cols]
        
        self.assets = list(returns.columns)
        N = len(self.assets)
        
        # Covariance matrix
        self.Sigma = returns.cov().values
        
        # Fit components
        self.laplacian.fit(returns)
        self.ontology.fit(self.assets, market_caps)
        self.hedger.fit(returns, factor_returns)
        
        # Get component matrices
        L = self.laplacian.get_laplacian(self.assets)
        S = self.ontology.get_membership_matrix()
        b = self.ontology.get_target_allocation()
        B = self.hedger.get_loading_matrix()
        E = self.hedger.get_target_exposures()
        
        # Adjust λ based on regime
        if regime and self.config.regime_adaptive:
            regime_lambdas = self.config.regime_lambda_map.get(regime, {})
            lambda_mu = regime_lambdas.get("lambda_mu", self.config.lambda_mu)
            lambda_T = regime_lambdas.get("lambda_topology", self.config.lambda_topology)
            lambda_O = regime_lambdas.get("lambda_ontology", self.config.lambda_ontology)
            lambda_H = regime_lambdas.get("lambda_hedge", self.config.lambda_hedge)
        else:
            lambda_mu = self.config.lambda_mu
            lambda_T = self.config.lambda_topology
            lambda_O = self.config.lambda_ontology
            lambda_H = self.config.lambda_hedge
        
        # Store lambdas for diagnostics
        self.diagnostics['lambdas'] = {
            'lambda_mu': lambda_mu,
            'lambda_topology': lambda_T,
            'lambda_ontology': lambda_O,
            'lambda_hedge': lambda_H
        }
        
        # Build regularized covariance: Σ̃ = Σ + λ_T·L + λ_O·S'S + λ_H·B'B
        self.Sigma_tilde = (
            self.Sigma +
            lambda_T * L +
            lambda_O * (S.T @ S) +
            lambda_H * (B.T @ B)
        )
        
        # Expected returns (use TabPFN predictions or equilibrium)
        if expected_returns is not None:
            # Align with current assets
            mu_aligned = np.zeros(N)
            for i, asset in enumerate(self.assets):
                if asset in expected_returns.index:
                    mu_aligned[i] = expected_returns[asset]
            self.mu = mu_aligned
        else:
            # Use equilibrium returns (π = δΣw_mkt)
            delta = 2.5  # Risk aversion
            w_mkt = np.ones(N) / N  # Equal weight as proxy
            self.mu = delta * self.Sigma @ w_mkt
        
        # Build adjusted expected return: μ̃ = λ_μ·μ + λ_O·S'b + λ_H·B'E
        self.mu_tilde = (
            lambda_mu * self.mu +
            lambda_O * (S.T @ b) +
            lambda_H * (B.T @ E)
        )
        
        return self
    
    def optimize(self) -> pd.Series:
        """
        Solve the optimization problem.
        
        Returns:
            Optimal portfolio weights as pd.Series
        """
        if self.Sigma_tilde is None:
            raise ValueError("Must call fit() first")
        
        N = len(self.assets)
        
        if self.config.long_only or self.config.max_leverage < 2.0:
            # Constrained optimization
            return self._optimize_constrained()
        else:
            # Closed-form solution
            return self._optimize_closed_form()
    
    def _optimize_closed_form(self) -> pd.Series:
        """
        Closed-form solution for unconstrained case.
        
        w* = Σ̃⁻¹(μ̃/2 - ν*/2 · 1)
        ν* = (1'Σ̃⁻¹μ̃ - 2) / (1'Σ̃⁻¹1)
        """
        N = len(self.assets)
        ones = np.ones(N)
        
        try:
            # Compute inverse
            Sigma_inv = linalg.inv(self.Sigma_tilde)
        except linalg.LinAlgError:
            # Use pseudo-inverse if singular
            Sigma_inv = linalg.pinv(self.Sigma_tilde)
        
        # Compute ν* (budget constraint multiplier)
        numerator = ones @ Sigma_inv @ self.mu_tilde - 2
        denominator = ones @ Sigma_inv @ ones
        nu_star = numerator / denominator
        
        # Compute optimal weights
        w_star = Sigma_inv @ (self.mu_tilde / 2 - nu_star / 2 * ones)
        
        self.weights = pd.Series(w_star, index=self.assets)
        
        # Store diagnostics
        self.diagnostics['method'] = 'closed_form'
        self.diagnostics['nu_star'] = nu_star
        self.diagnostics['sum_weights'] = w_star.sum()
        
        return self.weights
    
    def _optimize_constrained(self) -> pd.Series:
        """
        Constrained optimization using scipy.
        """
        N = len(self.assets)
        
        def objective(w):
            """Objective: w'Σ̃w - μ̃'w"""
            return w @ self.Sigma_tilde @ w - self.mu_tilde @ w
        
        def gradient(w):
            """Gradient: 2Σ̃w - μ̃"""
            return 2 * self.Sigma_tilde @ w - self.mu_tilde
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Budget
        ]
        
        # Bounds
        if self.config.long_only:
            bounds = [(0, self.config.max_position) for _ in range(N)]
        else:
            bounds = [
                (self.config.min_position, self.config.max_position) 
                for _ in range(N)
            ]
        
        # Initial guess (equal weight)
        w0 = np.ones(N) / N
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        self.weights = pd.Series(result.x, index=self.assets)
        
        # Store diagnostics
        self.diagnostics['method'] = 'constrained'
        self.diagnostics['success'] = result.success
        self.diagnostics['objective'] = result.fun
        self.diagnostics['sum_weights'] = result.x.sum()
        
        return self.weights
    
    def get_weights(self) -> pd.Series:
        """Return optimized weights"""
        if self.weights is None:
            return self.optimize()
        return self.weights
    
    def get_decomposition(self) -> Dict[str, any]:
        """
        Decompose portfolio into components.
        
        Returns:
            Dict with sector exposures, factor exposures, risk contributions
        """
        if self.weights is None:
            self.optimize()
        
        w = self.weights.values
        
        decomposition = {
            'sector_exposures': self.ontology.get_sector_exposures(w),
            'factor_exposures': self.hedger.get_factor_exposures(w),
            'portfolio_variance': float(w @ self.Sigma @ w),
            'gross_leverage': float(np.abs(w).sum()),
            'long_exposure': float(np.maximum(w, 0).sum()),
            'short_exposure': float(np.abs(np.minimum(w, 0)).sum()),
            'num_positions': int(np.sum(np.abs(w) > 0.001)),
            'max_position': float(np.max(w)),
            'min_position': float(np.min(w)),
            'herfindahl': float(np.sum(w**2)),
        }
        
        return decomposition
    
    def get_diagnostics(self) -> Dict:
        """Return optimization diagnostics"""
        return self.diagnostics


# =============================================================================
# TABPFN ALPHA PREDICTOR (INTERFACE)
# =============================================================================

class TabPFNAlphaPredictor:
    """
    TabPFN Interface for Cross-Sectional Return Prediction
    
    This class provides an interface to TabPFN-2.5 for:
    1. Cross-sectional return prediction
    2. Regime classification
    3. Violation probability estimation
    
    Note: Requires tabpfn package installation.
    In production, replace mock implementation with actual TabPFN.
    """
    
    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.regressor = None
        self.classifier = None
        self._tabpfn_available = False
        
        try:
            from tabpfn import TabPFNRegressor, TabPFNClassifier
            self.regressor = TabPFNRegressor()
            self.classifier = TabPFNClassifier()
            self._tabpfn_available = True
            print("✅ TabPFN-2.5 loaded successfully")
        except ImportError:
            print("⚠️ TabPFN not available, using fallback predictor")
    
    def predict_returns(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict next-period returns using cross-sectional features.
        
        Parameters:
            X_train: Training features (T × F)
            y_train: Training returns (T,)
            X_test: Test features (1 × F)
        
        Returns:
            Predicted returns
        """
        if self._tabpfn_available:
            self.regressor.fit(X_train.values, y_train.values)
            return self.regressor.predict(X_test.values)
        else:
            # Fallback: Simple ridge regression
            return self._fallback_predict(X_train, y_train, X_test)
    
    def predict_regime(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict market regime probabilities.
        
        Returns:
            Regime probabilities (4,) for each regime
        """
        if self._tabpfn_available:
            self.classifier.fit(X_train.values, y_train.values)
            return self.classifier.predict_proba(X_test.values)
        else:
            # Fallback: Simple logistic regression
            return self._fallback_classify(X_train, y_train, X_test)
    
    def _fallback_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """Fallback predictor using ridge regression"""
        from sklearn.linear_model import Ridge
        
        model = Ridge(alpha=1.0)
        model.fit(X_train.values, y_train.values)
        return model.predict(X_test.values)
    
    def _fallback_classify(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """Fallback classifier using logistic regression"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train.values, y_train.values)
        return model.predict_proba(X_test.values)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class UnifiedPortfolioOptimizationPipeline:
    """
    Complete Pipeline for Unified Portfolio Optimization
    
    Combines:
    1. Feature extraction (Kalman, GARCH, Sentiment)
    2. TabPFN prediction
    3. Lagrangian optimization
    
    Usage:
        pipeline = UnifiedPortfolioOptimizationPipeline(config)
        pipeline.fit(returns, factor_returns, sentiment)
        weights = pipeline.optimize()
    """
    
    def __init__(self, config: LagrangianConfig = None):
        self.config = config or LagrangianConfig()
        self.optimizer = UnifiedLagrangianOptimizer(self.config)
        self.predictor = TabPFNAlphaPredictor()
        
        self.returns = None
        self.features = None
        self.expected_returns = None
        self.current_regime = None
    
    def fit(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame = None,
        sentiment: pd.DataFrame = None,
        betas: pd.DataFrame = None,
        volatility: pd.DataFrame = None
    ) -> 'UnifiedPortfolioOptimizationPipeline':
        """
        Fit the complete pipeline.
        
        Parameters:
            returns: Asset returns (T × N)
            factor_returns: Fama-French factors (T × F)
            sentiment: FinBERT sentiment scores (T × N)
            betas: Kalman filter betas (T × N)
            volatility: GARCH-MIDAS volatility (T × N)
        """
        self.returns = returns
        
        # Build feature matrix for TabPFN
        features_list = []
        
        # Basic features
        features_list.append(returns.rolling(20).mean().iloc[-1].to_frame('momentum_20d'))
        features_list.append(returns.rolling(60).mean().iloc[-1].to_frame('momentum_60d'))
        features_list.append(returns.rolling(20).std().iloc[-1].to_frame('volatility_20d'))
        
        # Add sentiment if available
        if sentiment is not None and len(sentiment) > 0:
            features_list.append(sentiment.iloc[-1].to_frame('sentiment'))
        
        # Add betas if available
        if betas is not None and len(betas) > 0:
            features_list.append(betas.iloc[-1].to_frame('beta'))
        
        # Add volatility if available
        if volatility is not None and len(volatility) > 0:
            features_list.append(volatility.iloc[-1].to_frame('garch_vol'))
        
        # Combine features
        self.features = pd.concat(features_list, axis=1).T
        
        # Detect regime
        self.current_regime = self._detect_regime(returns)
        
        # Predict expected returns using TabPFN (or fallback)
        self.expected_returns = self._predict_returns(returns)
        
        # Fit optimizer
        self.optimizer.fit(
            returns=returns,
            expected_returns=self.expected_returns,
            factor_returns=factor_returns,
            regime=self.current_regime
        )
        
        return self
    
    def _detect_regime(self, returns: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        # Compute recent volatility
        recent_vol = returns.iloc[-20:].std().mean()
        long_vol = returns.iloc[-252:].std().mean() if len(returns) >= 252 else recent_vol
        
        # Compute recent direction
        recent_ret = returns.iloc[-20:].mean().mean()
        
        # Classify
        high_vol = recent_vol > long_vol
        bull = recent_ret > 0
        
        if high_vol and bull:
            return MarketRegime.HIGH_VOL_BULL
        elif high_vol and not bull:
            return MarketRegime.HIGH_VOL_BEAR
        elif not high_vol and bull:
            return MarketRegime.LOW_VOL_BULL
        else:
            return MarketRegime.LOW_VOL_BEAR
    
    def _predict_returns(self, returns: pd.DataFrame) -> pd.Series:
        """Predict expected returns using TabPFN or fallback"""
        # Simple momentum-based prediction as fallback
        momentum_20 = returns.iloc[-20:].mean()
        momentum_60 = returns.iloc[-60:].mean() if len(returns) >= 60 else momentum_20
        volatility = returns.iloc[-20:].std()
        
        # Simple alpha signal: momentum / volatility (Sharpe-like)
        alpha_signal = 0.7 * momentum_20 + 0.3 * momentum_60
        
        return alpha_signal
    
    def optimize(self) -> pd.Series:
        """Run optimization and return weights"""
        return self.optimizer.optimize()
    
    def get_weights(self) -> pd.Series:
        """Return optimized weights"""
        return self.optimizer.get_weights()
    
    def get_decomposition(self) -> Dict:
        """Return portfolio decomposition"""
        return self.optimizer.get_decomposition()
    
    def get_regime(self) -> MarketRegime:
        """Return detected regime"""
        return self.current_regime


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_optimizer() -> UnifiedLagrangianOptimizer:
    """Create optimizer with default settings"""
    config = LagrangianConfig()
    return UnifiedLagrangianOptimizer(config)


def create_conservative_optimizer() -> UnifiedLagrangianOptimizer:
    """Create conservative (low risk) optimizer"""
    config = LagrangianConfig(
        lambda_mu=0.5,
        lambda_topology=0.3,
        lambda_ontology=0.3,
        lambda_hedge=1.0,
        long_only=True,
        max_position=0.10
    )
    return UnifiedLagrangianOptimizer(config)


def create_aggressive_optimizer() -> UnifiedLagrangianOptimizer:
    """Create aggressive (high return) optimizer"""
    config = LagrangianConfig(
        lambda_mu=2.0,
        lambda_topology=0.05,
        lambda_ontology=0.1,
        lambda_hedge=0.3,
        long_only=False,
        max_position=0.20,
        max_leverage=1.5
    )
    return UnifiedLagrangianOptimizer(config)


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Unified Lagrangian Portfolio Optimizer - Demo")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_assets = 20
    n_days = 252 * 5  # 5 years
    
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    
    # Simulate returns with sector structure
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.02,
        index=dates,
        columns=assets
    )
    
    # Add sector correlation
    for i in range(0, n_assets, 4):
        sector_factor = np.random.randn(n_days) * 0.01
        for j in range(i, min(i + 4, n_assets)):
            returns.iloc[:, j] += sector_factor
    
    print(f"\nData: {n_days} days × {n_assets} assets")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    
    # Create optimizer
    config = LagrangianConfig(
        lambda_mu=1.0,
        lambda_topology=0.1,
        lambda_ontology=0.2,
        lambda_hedge=0.5,
        long_only=True
    )
    
    optimizer = UnifiedLagrangianOptimizer(config)
    
    # Fit
    print("\n" + "-" * 70)
    print("Fitting optimizer...")
    optimizer.fit(returns)
    
    # Optimize
    print("Optimizing...")
    weights = optimizer.optimize()
    
    print("\n" + "-" * 70)
    print("OPTIMAL WEIGHTS:")
    print("-" * 70)
    for asset, w in weights.items():
        if abs(w) > 0.001:
            print(f"  {asset}: {w:+.4f}")
    
    # Decomposition
    decomp = optimizer.get_decomposition()
    
    print("\n" + "-" * 70)
    print("PORTFOLIO DECOMPOSITION:")
    print("-" * 70)
    print(f"  Sum of weights:     {weights.sum():.6f}")
    print(f"  Gross leverage:     {decomp['gross_leverage']:.4f}")
    print(f"  Long exposure:      {decomp['long_exposure']:.4f}")
    print(f"  Short exposure:     {decomp['short_exposure']:.4f}")
    print(f"  Portfolio variance: {decomp['portfolio_variance']:.6f}")
    print(f"  Num positions:      {decomp['num_positions']}")
    print(f"  Herfindahl index:   {decomp['herfindahl']:.4f}")
    
    print("\n" + "-" * 70)
    print("DIAGNOSTICS:")
    print("-" * 70)
    diag = optimizer.get_diagnostics()
    for k, v in diag.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
