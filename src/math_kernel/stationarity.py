import numpy as np
import pandas as pd

def get_weights(d, size):
    """
    Calcula los pesos de la diferenciación fraccionaria usando el método de expansión de binomio.
    
    Args:
        d (float): El orden de diferenciación (0 < d < 1).
        size (int): El número de pesos a calcular.
        
    Returns:
        np.array: Pesos calculados.
    """
    w = [1.0]
    for k in range(1, size):
        w_curr = -w[-1] * (d - k + 1) / k
        w.append(w_curr)
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Aplica diferenciación fraccionaria con ventana fija (Fixed Window Fractional Difference).
    A diferencia de la diferenciación entera (d=1), conserva la memoria de largo plazo.
    
    Args:
        series (pd.Series): La serie temporal original (usualmente log-prices).
        d (float): El orden de diferenciación (especificado en README como 0.4).
        thres (float): Umbral para descartar pesos insignificantes y ahorrar cómputo.
        
    Returns:
        pd.Series: La serie diferenciada fraccionalmente.
    """
    # 1. Calcular pesos hasta el umbral
    w = get_weights(d, len(series))
    w_cum = np.cumsum(abs(w))
    w_cum /= w_cum[-1]
    skip = (w_cum > thres).sum()
    w = w[len(w) - skip:]
    
    # 2. Aplicar pesos a la serie
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        res = pd.Series(index=series_f.index, dtype=float)
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series_f.loc[loc, name]):
                continue
            res.loc[loc] = np.dot(w.T, series_f.iloc[iloc - skip + 1 : iloc + 1])[0, 0]
        df[name] = res
    
    return pd.DataFrame(df)

if __name__ == "__main__":
    # Test simple con datos aleatorios
    data = pd.DataFrame(np.cumsum(np.random.randn(1000, 1)), columns=['BTC'])
    diff = frac_diff_ffd(data, d=0.4)
    print("Serie Diferenciada (Primeras 5 líneas):")
    print(diff.head())
    print("\nEstadísticas de la serie fraccional:")
    print(diff.describe())
