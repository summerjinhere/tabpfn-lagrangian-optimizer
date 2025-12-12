# Unified Lagrangian Portfolio Optimizer

## 📋 Overview

A novel portfolio optimization framework that unifies:

1. **TabPFN-2.5** - Cross-sectional return prediction
2. **Topology (Graph Laplacian)** - Correlation-based diversification
3. **Ontology (Sector Constraints)** - GICS sector allocation
4. **Hedging (Factor Neutralization)** - FF5 + Momentum exposure control

All components are integrated via a **single Lagrangian optimization** framework.

## 🧮 Mathematical Framework

### Unified Lagrangian

```
L(w, λ) = w'Σ̃w - μ̃'w + ν(1'w - 1)
```

Where:
- **Σ̃** = Σ + λ_T·L + λ_O·S'S + λ_H·B'B  (Regularized Covariance)
- **μ̃** = λ_μ·μ + λ_O·S'b + λ_H·B'E  (Adjusted Expected Return)

### Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| Covariance | Σ | Historical covariance matrix |
| Topology | L | Graph Laplacian (diversification) |
| Ontology | S | Sector membership matrix |
| Hedging | B | Factor loading matrix |
| Alpha | μ | TabPFN predicted returns |

### Closed-Form Solution

```
w* = Σ̃⁻¹(μ̃/2 - ν*/2 · 1)

where ν* = (1'Σ̃⁻¹μ̃ - 2) / (1'Σ̃⁻¹1)
```

## 📁 Project Structure

```
ai_finance_v3/
├── unified_lagrangian_optimizer.py  # Core optimization engine
├── backtest_engine.py               # Rolling window backtester
├── feature_engineering.py           # Kalman, GARCH, Sentiment
├── data_loader.py                   # Yahoo/FF data loading
├── main.py                          # Main runner
└── README.md                        # This file
```

## 🚀 Quick Start

### Demo Mode

```bash
python main.py --mode demo
```

### Full Backtest

```bash
python main.py --mode backtest --universe diversified --start 2015-01-01
```

### Strategy Comparison

```bash
python main.py --mode compare
```

## ⚙️ Configuration

### Lagrangian Parameters

```python
config = LagrangianConfig(
    lambda_mu=1.0,        # Alpha signal trust
    lambda_topology=0.1,  # Diversification strength
    lambda_ontology=0.2,  # Sector adherence
    lambda_hedge=0.5,     # Factor neutralization
    long_only=True,       # No short selling
    max_position=0.15,    # Max 15% per asset
)
```

### Regime-Adaptive λ

| Regime | λ_μ | λ_T | λ_O | λ_H |
|--------|-----|-----|-----|-----|
| Low Vol Bull | 1.5 | 0.1 | 0.1 | 0.3 |
| Low Vol Bear | 0.8 | 0.3 | 0.3 | 1.0 |
| High Vol Bull | 1.0 | 0.2 | 0.2 | 0.8 |
| High Vol Bear | 0.5 | 0.5 | 0.4 | 1.5 |

## 📊 Features

### Feature Engineering

- **Kalman Filter Beta**: Time-varying market beta
- **GARCH-MIDAS Volatility**: Short-term + long-term volatility
- **Momentum**: 5d, 20d, 60d, 252d returns
- **Volatility-Adjusted Momentum**: Sharpe-like signals
- **Reversal**: Short-term vs long-term performance

### Topology (Graph Laplacian)

Converts correlation to distance matrix, then builds adjacency graph:

```python
d_ij = √(0.5 * (1 - ρ_ij))  # Distance
A_ij = exp(-d²_ij / σ²)      # Adjacency
L = D - A                    # Laplacian
```

### Ontology (Sector Constraints)

GICS sector classification with target allocations:
- Technology, Healthcare, Financials
- Consumer, Industrials, Energy
- Utilities, Materials, Real Estate

### Factor Hedging

FF5 + Momentum factor neutralization:
- Mkt-RF, SMB, HML, RMW, CMA, Mom

## 📈 Performance Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Information Ratio**: Alpha per unit tracking error
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / Max Drawdown
- **FF5 Alpha**: Risk-adjusted excess return with t-stat

## 📚 Academic References

1. **TabPFN**: Hollmann et al. (2025) "Accurate predictions on small data with a tabular foundation model" - *Nature*

2. **Network Risk Parity**: Ciciretti & Pallotta (2024) "Graph theory-based portfolio construction" - *J. Asset Management*

3. **FinBERT + BL**: Colasanto et al. (2022) "BERT's sentiment score for portfolio optimization" - *Annals of Operations Research*

4. **HRP**: Lopez de Prado (2016) "Building Diversified Portfolios that Outperform Out of Sample" - *J. Portfolio Management*

5. **Augmented Lagrangian**: Enkhsaikhan & Jo (2025) "Risk-constrained RL for Portfolio Optimization" - *AAAI*

## 🔧 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
statsmodels>=0.13.0
yfinance>=0.1.70 (optional)
pandas-datareader>=0.10.0 (optional)
tabpfn>=2.5.0 (optional)
scikit-learn>=1.0.0
```

## 📄 License

Academic use only. For commercial use, contact authors.

## 👤 Author

AI Finance Research Team
Sogang University, M.A. AI-Finance Program
December 2025
