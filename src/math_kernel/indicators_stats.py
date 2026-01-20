import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

def compute_arima_forecast(series, order=(1, 1, 1)):
    """
    Calcula el pronóstico de precios usando un modelo ARIMA(5,1,0).
    Captura las dependencias lineales y la autocorrelación en la serie.
    """
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        return forecast
    except:
        return np.nan

def compute_garch_volatility(series, p=1, q=1):
    """
    Calcula la volatilidad condicional usando un modelo GARCH(1,1).
    Util para modelar la "agrupación de volatilidad" (volatility clustering).
    """
    try:
        # Escalar series para mejorar convergencia
        scaling_factor = 100
        am = arch_model(series * scaling_factor, vol='Garch', p=p, q=q, dist='normal')
        res = am.fit(disp='off')
        vol = res.conditional_volatility[-1] / scaling_factor
        return vol
    except:
        return np.nan

def compute_egarch_volatility(series, p=1, q=1):
    """
    Calcula la volatilidad usando EGARCH, que captura la asimetría 
    (efecto apalancamiento: las caídas suelen generar más volatilidad que las alzas).
    """
    try:
        scaling_factor = 100
        am = arch_model(series * scaling_factor, vol='EGARCH', p=p, q=q, dist='normal')
        res = am.fit(disp='off')
        vol = res.conditional_volatility[-1] / scaling_factor
        return vol
    except:
        return np.nan

def process_stats_indicators(df_frac):
    """
    Procesa los indicadores estadísticos para un DataFrame de series fraccionalmente diferenciadas.
    """
    results = {}
    for col in df_frac.columns:
        series = df_frac[col].dropna()
        if len(series) < 50: # Mínimo de datos para modelos robustos
            continue
            
        results[f"{col}_arima"] = compute_arima_forecast(series)
        results[f"{col}_garch"] = compute_garch_volatility(series)
        results[f"{col}_egarch"] = compute_egarch_volatility(series)
        
    return results

if __name__ == "__main__":
    # Test con datos sintéticos (100 puntos de retorno fraccional)
    returns = np.random.normal(0, 0.01, 100)
    df_test = pd.DataFrame({'BTC': returns})
    
    print("Calculando indicadores estadísticos core...")
    stats = process_stats_indicators(df_test)
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")
