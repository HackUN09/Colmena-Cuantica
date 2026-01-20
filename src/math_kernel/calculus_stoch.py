import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

def apply_ito_lemma(price_series, dt=1/252):
    """
    Aproximación del Lema de Itô para la evolución de una función de precio.
    df = (df/dt + mu*S*df/dS + 0.5*sigma^2*S^2*d^2f/dS^2)dt + sigma*S*df/dS*dW
    """
    try:
        S = price_series.iloc[-1]
        returns = price_series.pct_change().dropna()
        mu = returns.mean() / dt
        sigma = returns.std() / np.sqrt(dt)
        
        # Deriva del proceso
        drift = (mu * S) * dt
        # Difusión (volatilidad)
        diffusion = (sigma * S) * np.sqrt(dt)
        
        return {'drift': drift, 'diffusion': diffusion}
    except:
        return None

def fit_ornstein_uhlenbeck(series):
    """
    Ajusta un proceso Ornstein-Uhlenbeck: dXt = lambda(mu - Xt)dt + sigma*dWt
    Provee el nivel de reversión a la media (mu) y la velocidad de reversión (lambda).
    """
    try:
        y = series.diff().dropna()
        x = series.shift(1).dropna()
        x = x[y.index] # Alinear
        
        # Regresión: dy = a + b*x + e
        # lambda = -b / dt, mu = -a / b
        import statsmodels.api as sm
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        
        a, b = model.params
        lambda_val = -b
        mu_val = -a / b
        
        return {
            'reversion_speed': lambda_val,
            'mean_level': mu_val,
            'is_mean_reverting': lambda_val > 0
        }
    except:
        return None

if __name__ == "__main__":
    # Test con serie de reversión a la media sintética
    np.random.seed(42)
    s = [100]
    for _ in range(200):
        s.append(s[-1] + 0.5 * (100 - s[-1]) + np.random.normal(0, 1))
    
    series = pd.Series(s)
    print("Ajustando Proceso Ornstein-Uhlenbeck...")
    print(fit_ornstein_uhlenbeck(series))
