import numpy as np
import pandas as pd

def compute_gradients(series):
    """
    Calcula la 1ra derivada (Momentum/Velocidad) y 2da derivada (Aceleración/Convexidad).
    Permite detectar el agotamiento de una tendencia antes de su reversión.
    """
    try:
        # Velocidad: Diferencia primera
        velocity = series.diff()
        # Aceleración: Cambio en la velocidad
        acceleration = velocity.diff()
        
        return {
            'velocity': velocity.iloc[-1],
            'acceleration': acceleration.iloc[-1],
            'is_convex': acceleration.iloc[-1] > 0
        }
    except:
        return None

def compute_portfolio_jacobian(returns_df):
    """
    Calcula la Matriz Jacobiana de sensibilidad.
    Mide cómo cambia el retorno del portafolio ante cambios marginales en cada activo.
    """
    try:
        # Matriz de covarianza como proxy de sensibilidad conjunta
        cov_matrix = returns_df.cov().values
        # Jacobiana representa el gradiente de f(x) respecto a x
        # En este contexto, es la matriz de sensibilidades parciales
        return cov_matrix # En un modelo lineal, la Jacobiana son los coeficientes
    except:
        return None

def compute_hessian_inflection(series):
    """
    Usa la Matriz Hessiana (segunda derivada masiva) para encontrar puntos de inflexión crítica.
    """
    try:
        # Aproximación numérica de la Hessiana en series temporales
        grad = np.gradient(series.dropna())
        hess = np.gradient(grad)
        return hess[-1]
    except:
        return np.nan

if __name__ == "__main__":
    # Test
    data = pd.Series(np.sin(np.linspace(0, 10, 100)))
    print("Calculando Gradientes...")
    print(compute_gradients(data))
    print(f"Hessiana (Punto de inflexión): {compute_hessian_inflection(data):.6f}")
