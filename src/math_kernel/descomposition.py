from statsmodels.tsa.seasonal import STL
import pandas as pd
import matplotlib.pyplot as plt

def decompose_stl(series, period=None):
    """
    Descompone la serie usando STL (Seasonal-Trend decomposition using LOESS).
    Permite aislar el 'ruido' (residuo) que suele ser el foco de los modelos de caos.
    """
    try:
        # Si no se especifica periodo, intentamos inferir o usamos uno estándar para cripto
        # (ej. 24 para datos horarios si fuera el caso)
        stl = STL(series, period=period, robust=True)
        res = stl.fit()
        return {
            'trend': res.trend,
            'seasonal': res.seasonal,
            'resid': res.resid
        }
    except:
        return None

if __name__ == "__main__":
    import numpy as np
    t = np.linspace(0, 100, 500)
    # Tendencia + Estacionalidad + Ruido
    s = 0.1 * t + np.sin(2 * np.pi * t / 10) + np.random.normal(0, 0.5, 500)
    series = pd.Series(s)
    
    print("Ejecutando descomposición STL...")
    decomp = decompose_stl(series, period=10)
    if decomp:
        print(" STL completado con éxito.")
        print(f"Media del residuo: {decomp['resid'].mean():.6f}")
