"""
Data Loader for Unified Lagrangian Portfolio
=============================================

Provides:
1. Yahoo Finance data fetching
2. Fama-French factor data
3. Data preprocessing and alignment

Author: AI Finance Research
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# YAHOO FINANCE DATA LOADER
# =============================================================================

class YahooDataLoader:
    """
    Load stock data from Yahoo Finance.
    
    Handles:
    - Price data (OHLCV)
    - Returns calculation
    - Missing data handling
    """
    
    def __init__(self, cache_dir: str = "."):
        self.cache_dir = cache_dir
        self._yf_available = False
        
        try:
            import yfinance as yf
            self._yf = yf
            self._yf_available = True
        except ImportError:
            print("⚠️ yfinance not available. Install with: pip install yfinance")
    
    def load_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load adjusted close prices for given tickers.
        
        Parameters:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (default: today)
        
        Returns:
            DataFrame of adjusted close prices (T × N)
        """
        if not self._yf_available:
            return self._generate_dummy_prices(tickers, start_date, end_date)
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Loading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        try:
            data = self._yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                prices = data[['Adj Close']]
                prices.columns = [tickers[0]]
            
            print(f"✅ Loaded {len(prices)} days of data")
            return prices
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self._generate_dummy_prices(tickers, start_date, end_date)
    
    def load_returns(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """Load and calculate returns."""
        prices = self.load_prices(tickers, start_date, end_date)
        returns = prices.pct_change().dropna()
        return returns
    
    def _generate_dummy_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """Generate dummy price data for testing."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        dates = pd.date_range(start_date, end_date, freq='B')
        n_days = len(dates)
        
        # Generate correlated returns
        np.random.seed(42)
        n_assets = len(tickers)
        
        # Create correlation structure
        corr = np.eye(n_assets) * 0.5 + 0.5
        chol = np.linalg.cholesky(corr)
        
        # Generate returns
        raw_returns = np.random.randn(n_days, n_assets) @ chol.T * 0.02
        
        # Convert to prices
        prices = np.exp(np.cumsum(raw_returns, axis=0)) * 100
        
        return pd.DataFrame(prices, index=dates, columns=tickers)


# =============================================================================
# FAMA-FRENCH FACTOR DATA
# =============================================================================

class FamaFrenchLoader:
    """
    Load Fama-French factor data.
    
    Factors:
    - Mkt-RF: Market excess return
    - SMB: Small Minus Big
    - HML: High Minus Low
    - RMW: Robust Minus Weak
    - CMA: Conservative Minus Aggressive
    - Mom: Momentum
    """
    
    # Kenneth French data library URLs
    FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
    
    def __init__(self):
        self._pdr_available = False
        
        try:
            import pandas_datareader.data as pdr
            self._pdr = pdr
            self._pdr_available = True
        except ImportError:
            print("⚠️ pandas_datareader not available")
    
    def load_ff5_plus_momentum(
        self,
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load Fama-French 5 factors plus Momentum.
        
        Returns:
            DataFrame with columns: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'RF']
        """
        if not self._pdr_available:
            return self._generate_dummy_factors(start_date, end_date)
        
        try:
            # Load FF5
            ff5 = self._pdr.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start=start_date,
                end=end_date
            )[0]
            
            # Load Momentum
            mom = self._pdr.DataReader(
                'F-F_Momentum_Factor_daily',
                'famafrench',
                start=start_date,
                end=end_date
            )[0]
            
            # Combine
            factors = ff5.join(mom)
            factors.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
            
            # Convert from percentage to decimal
            factors = factors / 100
            
            print(f"✅ Loaded {len(factors)} days of factor data")
            return factors
            
        except Exception as e:
            print(f"❌ Error loading factor data: {e}")
            return self._generate_dummy_factors(start_date, end_date)
    
    def _generate_dummy_factors(
        self,
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """Generate dummy factor data for testing."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        dates = pd.date_range(start_date, end_date, freq='B')
        n_days = len(dates)
        
        np.random.seed(42)
        
        factors = pd.DataFrame({
            'Mkt-RF': np.random.randn(n_days) * 0.01,
            'SMB': np.random.randn(n_days) * 0.005,
            'HML': np.random.randn(n_days) * 0.005,
            'RMW': np.random.randn(n_days) * 0.003,
            'CMA': np.random.randn(n_days) * 0.003,
            'Mom': np.random.randn(n_days) * 0.008,
            'RF': np.ones(n_days) * 0.0001,  # ~2.5% annual
        }, index=dates)
        
        return factors


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """
    Data preprocessing and alignment.
    
    Handles:
    - Missing value imputation
    - Outlier treatment
    - Data alignment
    - Split into train/test
    """
    
    def __init__(
        self,
        max_missing_pct: float = 0.1,
        outlier_std: float = 5.0
    ):
        self.max_missing_pct = max_missing_pct
        self.outlier_std = outlier_std
    
    def clean_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Clean return data.
        
        Steps:
        1. Remove columns with too many missing values
        2. Fill remaining missing values
        3. Winsorize outliers
        """
        # Remove high-missing columns
        missing_pct = returns.isna().mean()
        valid_cols = missing_pct[missing_pct <= self.max_missing_pct].index
        returns = returns[valid_cols]
        
        # Fill missing (forward then backward)
        returns = returns.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Winsorize outliers
        mean = returns.mean()
        std = returns.std()
        lower = mean - self.outlier_std * std
        upper = mean + self.outlier_std * std
        
        returns = returns.clip(lower=lower, upper=upper, axis=1)
        
        return returns
    
    def align_data(
        self,
        returns: pd.DataFrame,
        factors: pd.DataFrame = None,
        sentiment: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Align all data to common dates."""
        common_idx = returns.index
        
        if factors is not None:
            common_idx = common_idx.intersection(factors.index)
        if sentiment is not None:
            common_idx = common_idx.intersection(sentiment.index)
        
        returns = returns.loc[common_idx]
        
        if factors is not None:
            factors = factors.loc[common_idx]
        if sentiment is not None:
            sentiment = sentiment.loc[common_idx]
        
        return returns, factors, sentiment
    
    def train_test_split(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        n = len(data)
        train_end = int(n * train_ratio)
        
        train = data.iloc[:train_end]
        test = data.iloc[train_end:]
        
        return train, test


# =============================================================================
# TICKER UNIVERSE
# =============================================================================

class TickerUniverse:
    """Pre-defined ticker universes."""
    
    # S&P 500 Top 50 (by market cap, approximate)
    SP500_TOP50 = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "UNH",
        "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "LLY", "PEP", "KO", "COST", "AVGO", "MCD", "WMT", "TMO", "CSCO",
        "ABT", "CRM", "ACN", "DHR", "ADBE", "NFLX", "PFE", "CMCSA", "AMD",
        "TXN", "NEE", "UPS", "PM", "INTC", "HON", "UNP", "LOW", "RTX",
        "QCOM", "BA", "CAT", "SPGI", "IBM"
    ]
    
    # Tech focused
    TECH_LEADERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "AVGO", "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM",
        "TXN", "MU", "NOW", "ORCL", "IBM", "AMAT"
    ]
    
    # Diversified sectors
    DIVERSIFIED = [
        # Tech
        "AAPL", "MSFT", "NVDA", "GOOGL",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV",
        # Financials
        "JPM", "BAC", "V", "MA",
        # Consumer
        "AMZN", "HD", "MCD", "NKE",
        # Energy
        "XOM", "CVX", "COP",
        # Industrials
        "BA", "CAT", "HON",
    ]
    
    @classmethod
    def get_universe(cls, name: str) -> List[str]:
        """Get ticker list by universe name."""
        universes = {
            "sp500_top50": cls.SP500_TOP50,
            "tech": cls.TECH_LEADERS,
            "diversified": cls.DIVERSIFIED,
        }
        return universes.get(name.lower(), cls.DIVERSIFIED)


# =============================================================================
# INTEGRATED DATA MANAGER
# =============================================================================

class DataManager:
    """
    Integrated Data Manager
    
    Provides unified interface for:
    - Loading price/return data
    - Loading factor data
    - Preprocessing
    - Caching
    """
    
    def __init__(self):
        self.price_loader = YahooDataLoader()
        self.factor_loader = FamaFrenchLoader()
        self.preprocessor = DataPreprocessor()
        
        self.returns = None
        self.factors = None
        self.prices = None
    
    def load_data(
        self,
        tickers: List[str] = None,
        universe: str = None,
        start_date: str = "2005-01-01",
        end_date: str = None
    ) -> Dict:
        """
        Load and preprocess all data.
        
        Parameters:
            tickers: List of ticker symbols
            universe: Pre-defined universe name
            start_date: Start date
            end_date: End date
        
        Returns:
            Dict with 'returns', 'factors', 'prices', 'market'
        """
        # Get tickers
        if tickers is None:
            if universe:
                tickers = TickerUniverse.get_universe(universe)
            else:
                tickers = TickerUniverse.DIVERSIFIED
        
        # Load prices and returns
        prices = self.price_loader.load_prices(tickers, start_date, end_date)
        returns = prices.pct_change().dropna()
        
        # Clean returns
        returns = self.preprocessor.clean_returns(returns)
        
        # Load factors
        factors = self.factor_loader.load_ff5_plus_momentum(start_date, end_date)
        
        # Align data
        returns, factors, _ = self.preprocessor.align_data(returns, factors)
        
        # Market returns
        market = factors['Mkt-RF'] + factors['RF'] if factors is not None else returns.mean(axis=1)
        
        self.returns = returns
        self.factors = factors
        self.prices = prices
        
        print(f"\n{'='*50}")
        print("DATA SUMMARY")
        print(f"{'='*50}")
        print(f"Tickers:     {len(returns.columns)}")
        print(f"Date range:  {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"Trading days: {len(returns)}")
        if factors is not None:
            print(f"Factors:     {list(factors.columns)}")
        
        return {
            'returns': returns,
            'factors': factors,
            'prices': prices,
            'market': market,
            'tickers': list(returns.columns),
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Data Loader - Demo")
    print("=" * 70)
    
    # Initialize
    dm = DataManager()
    
    # Load data
    data = dm.load_data(
        universe="diversified",
        start_date="2020-01-01"
    )
    
    print("\n" + "-" * 70)
    print("SAMPLE DATA")
    print("-" * 70)
    
    print("\nReturns (last 5 days):")
    print(data['returns'].tail())
    
    if data['factors'] is not None:
        print("\nFactors (last 5 days):")
        print(data['factors'].tail())
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
