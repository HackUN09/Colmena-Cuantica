"""
COLMENA-CUÁNTICA v12.0 - Scale Invariant Transformations
===========================================================
Implementa transformaciones que garantizan invarianza bajo cambios de escala.

Fundamento Matemático:
Una función f es scale-invariant si:
    f(k·x) = f(x)  ∀k > 0

Para precios financieros, esto se logra mediante:
1. Log-returns: log(P_t / P_{t-1}) invariante a P → k·P
2. Ratios: high/low invariante a multiplicación
3. Z-scores: (x - μ) / σ invariante a traslaciones

Teorema (Buckingham π):
Todo sistema físico puede expresarse en términos de cantidades adimensionales.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.stats import zscore


class ScaleInvariantTransform:
    """
    Aplicador de transformaciones scale-invariant a series financieras.
    
    Propiedades garantizadas:
    - Homegenidad: f(αx) = α^k f(x) para algún k
    - Adimensionalidad: output sin unidades físicas
    - Robustez: resistente a outliers (usa mediana)
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Ventana para estadísticas rodantes
        """
        self.window_size = window_size
        self.stats = {}  # Guardar stats para denormalize
        
    def log_returns(self, prices: pd.Series) -> pd.Series:
        """
        Retorna logarítmicos: r_t = log(P_t / P_(t-1))
        
        Propiedad: log(k·P_t / k·P_(t-1)) = log(P_t / P_(t-1))
        """
        returns = np.log(prices / prices.shift(1))
        return returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
    def robust_zscore(self, series: pd.Series, name: str = 'feature') -> pd.Series:
        """
        Z-score robusto usando mediana en vez de media.
        
        z = (x - median(x)) / IQR(x)
        
        donde IQR = Q3 - Q1 (Rango intercuartílico)
        """
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr < 1e-10:
            iqr = series.std() + 1e-10  # Fallback a std si IQR=0
            
        z = (series - median) / iqr
        
        # Guardar stats para denormalize
        self.stats[name] = {'median': median, 'iqr': iqr}
        
        return z.clip(-3, 3)  # Clip outliers
        
    def rolling_zscore(self, series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Z-score sobre ventana rodante (time-translation invariant)
        """
        w = window or self.window_size
        rolling_mean = series.rolling(window=w, min_periods=int(w/2)).mean()
        rolling_std = series.rolling(window=w, min_periods=int(w/2)).std()
        
        z = (series - rolling_mean) / (rolling_std + 1e-10)
        return z.fillna(0).clip(-3, 3)
        
    def dimensionless_ratio(self, numerator: pd.Series, denominator: pd.Series, name: str = 'ratio') -> pd.Series:
        """
        Ratio adimensional: x / y
        
        Invariante a: (k·x) / (k·y) = x / y
        """
        ratio = numerator / (denominator + 1e-10)
        return self.robust_zscore(ratio, name)
        
    def percent_change(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """
        Cambio porcentual: (P_t - P_(t-n)) / P_(t-n)
        
        Scale-invariant (adimensional)
        """
        pct = series.pct_change(periods=periods)
        return pct.replace([np.inf, -np.inf], np.nan).fillna(0)


def detect_optimal_resolution(
    price_series: pd.Series,
    resolutions: list = [1, 5, 15, 60]
) -> Tuple[int, float]:
    """
    Detecta la resolución temporal óptima usando autocorrelación parcial.
    
    Args:
        price_series: Serie de precios (resolución base 1-min)
        resolutions: Lista de resoluciones a probar (en minutos)
        
    Returns:
        (optimal_resolution_min, max_pacf)
        
    Fundamento:
    La resolución óptima maximiza la autocorrelación parcial,
    indicando dónde la memoria del proceso es más fuerte.
    """
    from statsmodels.tsa.stattools import pacf
    
    pacf_scores = {}
    
    for res in resolutions:
        # Downsample a resolución
        if res == 1:
            resampled = price_series
        else:
            # Simular resampling temporal
            resampled = price_series.iloc[::res]
            
        if len(resampled) < 50:
            continue
            
        log_ret = np.log(resampled / resampled.shift(1)).dropna()
        
        try:
            # Calcular PACF
            pacf_values = pacf(log_ret, nlags=10, method='ywm')
            # Suma de PACF significativos (>1.96/sqrt(N))
            threshold = 1.96 / np.sqrt(len(log_ret))
            score = np.sum(np.abs(pacf_values[1:]) > threshold)
            pacf_scores[res] = score
        except:
            pacf_scores[res] = 0
            
    if not pacf_scores:
        return 1, 0.0
        
    optimal = max(pacf_scores.items(), key=lambda x: x[1])
    return optimal[0], optimal[1]


def multi_resolution_features(
    df_ohlcv: pd.DataFrame,
    transformer: ScaleInvariantTransform,
    resolutions: list = [1, 5, 15]
) -> Dict[str, float]:
    """
    Calcula features en múltiples resoluciones temporales.
    
    Args:
        df_ohlcv: DataFrame con [open, high, low, close, volume]
        transformer: Instancia de ScaleInvariantTransform
        resolutions: Lista de minutos [1, 5, 15]
        
    Returns:
        Dict con features prefijadas por resolución
        
    Ejemplo:
        {'volatility_1m': 0.02, 'volatility_5m': 0.015, ...}
    """
    features = {}
    
    for res in resolutions:
        suffix = f"_{res}m"
        
        # Downsample
        if res == 1:
            df_res = df_ohlcv
        else:
            df_res = df_ohlcv.iloc[::res].copy()
            
        if len(df_res) < 10:
            continue
            
        # Calcular log-returns
        returns = transformer.log_returns(df_res['close'])
        
        # Features básicas
        features[f'volatility{suffix}'] = float(returns.std())
        features[f'skew{suffix}'] = float(returns.skew())
        features[f'kurtosis{suffix}'] = float(returns.kurtosis())
        
        # High-low ratio (invariante)
        hl_ratio = (df_res['high'] - df_res['low']) / (df_res['close'] + 1e-10)
        features[f'hl_ratio{suffix}'] = float(hl_ratio.mean())
        
    return features


def invariant_ratios(df_ohlcv: pd.DataFrame) -> Dict[str, float]:
    """
    Ratios adimensionales que son naturalmente scale-invariant.
    
    Estas cantidades NO cambian si multiplicamos todos los precios por k.
    """
    ratios = {}
    
    # 1. High-Low Spread normalizado
    hl_spread = (df_ohlcv['high'] - df_ohlcv['low']) / df_ohlcv['close']
    ratios['hl_spread_norm'] = float(hl_spread.mean())
    
    # 2. Open-Close ratio
    oc_ratio = df_ohlcv['open'] / (df_ohlcv['close'] + 1e-10)
    ratios['oc_ratio'] = float(oc_ratio.mean())
    
    # 3. Volume-Price correlation
    price_change = df_ohlcv['close'].pct_change().fillna(0)
    volume_change = df_ohlcv['volume'].pct_change().fillna(0)
    corr = price_change.corr(volume_change)
    ratios['volume_price_corr'] = float(corr) if not np.isnan(corr) else 0.0
    
    # 4. Amihud Illiquidity (inversamente proporcional a liquidez)
    # ILLIQ = |return| / volume
    returns = df_ohlcv['close'].pct_change().fillna(0)
    illiq = (returns.abs() / (df_ohlcv['volume'] + 1)).mean()
    ratios['amihud_illiquidity'] = float(illiq)
    
    return ratios


if __name__ == "__main__":
    # Tests
    print("=" * 70)
    print("Test: Scale Invariant Transformations")
    print("=" * 70)
    
    # Datos sintéticos
    np.random.seed(42)
    n = 200
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)  # Random walk
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(n) * 50,
        'high': prices + np.random.randn(n) * 50 + 50,
        'low': prices + np.random.randn(n) * 50 - 50,
        'close': prices,
        'volume': 1000 + np.random.randn(n) * 100
    })
    
    transformer = ScaleInvariantTransform()
    
    # Test 1: Log returns
    print("\n[TEST 1] Log Returns")
    returns = transformer.log_returns(df['close'])
    print(f"  Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    
    # Test 2: Robust Z-score
    print("\n[TEST 2] Robust Z-score")
    z = transformer.robust_zscore(df['close'], 'prices')
    print(f"  Median: {z.median():.6f}, IQR: {z.quantile(0.75) - z.quantile(0.25):.6f}")
    
    # Test 3: Optimal Resolution
    print("\n[TEST 3] Optimal Resolution Detection")
    opt_res, score = detect_optimal_resolution(df['close'])
    print(f"  Optimal resolution: {opt_res} minutes (score: {score})")
    
    # Test 4: Multi-resolution
    print("\n[TEST 4] Multi-resolution Features")
    multi_features = multi_resolution_features(df, transformer, [1, 5])
    for k, v in multi_features.items():
        print(f"  {k}: {v:.6f}")
    
    # Test 5: Invariant Ratios
    print("\n[TEST 5] Invariant Ratios")
    ratios = invariant_ratios(df)
    for k, v in ratios.items():
        print(f"  {k}: {v:.6f}")
    
    # Test 6: Scale Invariance Verification
    print("\n[TEST 6] Scale Invariance Verification")
    k = 10.0  # Multiplicador
    df_scaled = df.copy()
    df_scaled[['open', 'high', 'low', 'close']] *= k
    
    ratios_original = invariant_ratios(df)
    ratios_scaled = invariant_ratios(df_scaled)
    
    max_diff = max(abs(ratios_original[key] - ratios_scaled[key]) 
                   for key in ratios_original.keys())
    
    print(f"  Max difference after 10x scaling: {max_diff:.6f}")
    if max_diff < 0.01:
        print("  ✅ PASSED: Features are scale-invariant")
    else:
        print("  ❌ FAILED: Features changed significantly")
        
    print("\n✅ All tests completed")
