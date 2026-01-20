from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np

def run_adf_test(series):
    """
    Ejecuta el test de Dickey-Fuller Aumentada (ADF) para validar estacionariedad.
    Un p-value < 0.05 indica que la serie es estacionaria.
    """
    try:
        res = adfuller(series.dropna())
        return {
            'statistic': res[0],
            'p_value': res[1],
            'is_stationary': res[1] < 0.05
        }
    except:
        return None

def run_granger_causality(df, col_y, col_x, max_lag=5):
    """
    Determina si la serie X tiene valor predictivo sobre la serie Y.
    Crucial para detectar rotaciones de capital L1 -> DeFi, etc.
    """
    try:
        data = df[[col_y, col_x]].dropna()
        res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        # Retornamos el p-valor mínimo entre los lags para detectar cualquier relación
        p_values = [res[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
        return min(p_values)
    except:
        return 1.0

def run_johansen_cointegration(df_subset):
    """
    Test de Cointegración de Johansen.
    Busca relaciones de equilibrio de largo plazo entre un grupo de activos.
    """
    try:
        # det_order=0 (constante), k_ar_diff=1 (lag)
        res = coint_johansen(df_subset.dropna(), det_order=0, k_ar_diff=1)
        # Traza estadística vs valor crítico al 95%
        # Si trace_stat > cv_95, existe relación de cointegración
        return {
            'trace_stat': res.lr1,
            'crit_vals': res.cvt[:, 1] # 95% confidence
        }
    except:
        return None

if __name__ == "__main__":
    # Test
    data = pd.DataFrame({
        'BTC': np.random.randn(100).cumsum(),
        'ETH': np.random.randn(100).cumsum()
    })
    
    print("Ejecutando Test ADF para BTC...")
    print(run_adf_test(data['BTC']))
    
    print("\nEjecutando Causalidad de Granger (ETH -> BTC)...")
    print(f"P-Valor: {run_granger_causality(data, 'BTC', 'ETH')}")
