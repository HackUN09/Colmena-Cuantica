import numpy as np
import pandas as pd
from scipy.stats import entropy

def compute_hurst_exponent(series):
    """
    Calcula el Exponente de Hurst para medir la persistencia o memoria de una serie.
    H > 0.5: Persistente (tendencia).
    H < 0.5: Anti-persistente (reversión a la media).
    H = 0.5: Caminata aleatoria (Movimiento Browniano).
    """
    try:
        # Método R/S simplificado
        lags = range(2, 50)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return np.nan

def compute_shannon_entropy(series, bins=30):
    """
    Calcula la Entropía de Shannon para medir el desorden/incertidumbre en la señal.
    """
    try:
        hist, _ = np.histogram(series.dropna(), bins=bins, density=True)
        return entropy(hist + 1e-12) # Evitar log(0)
    except:
        return np.nan

def compute_lyapunov_exponent(series, delay=1, embed_dim=2):
    """
    Aproximación simplificada del Exponente de Lyapunov para detectar CAOS.
    Un exponente positivo indica sensibilidad extrema a las condiciones iniciales.
    """
    try:
        series = series.values
        # Cálculo de la divergencia de trayectorias cercanas
        # (Aproximación para series temporales de alta frecuencia)
        lags = np.arange(1, 10)
        divergence = []
        for lag in lags:
            d = np.abs(series[lag:] - series[:-lag])
            divergence.append(np.mean(np.log(d + 1e-12)))
        
        poly = np.polyfit(lags, divergence, 1)
        return poly[0] # Pendiente de la divergencia
    except:
        return np.nan

def compute_fractal_dimension(series):
    """
    Calcula la dimensión fractal de la serie (Método de Katz simplificado).
    Mide la complejidad de la 'curva' del precio.
    """
    try:
        y = series.values
        n = len(y) - 1
        L = np.sum(np.sqrt(1 + np.diff(y)**2)) # Longitud de la curva
        d = np.max(np.abs(y - y[0])) # Extensión de la curva
        return np.log(n) / (np.log(n) + np.log(d/L))
    except:
        return np.nan

if __name__ == "__main__":
    # Test
    t = np.linspace(0, 10, 1000)
    # Serie persistente
    persist = pd.Series(np.cumsum(np.random.normal(0.01, 0.01, 1000)))
    
    print("Métricas de Econofísica:")
    print(f"Hurst: {compute_hurst_exponent(persist):.4f}")
    print(f"Entropía: {compute_shannon_entropy(persist):.4f}")
    print(f"Lyapunov: {compute_lyapunov_exponent(persist):.4f}")
    print(f"Dim. Fractal: {compute_fractal_dimension(persist):.4f}")
