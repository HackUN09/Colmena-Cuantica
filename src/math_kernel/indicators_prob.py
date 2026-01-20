import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import genpareto, kstest

def compute_hmm_regimes(series, n_components=3):
    """
    Utiliza Modelos Ocultos de Markov (HMM) para detectar regímenes de mercado.
    Componentes típicos: 0=Bajista, 1=Neutral, 2=Alcista.
    """
    try:
        data = series.values.reshape(-1, 1)
        model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
        model.fit(data)
        states = model.predict(data)
        return states[-1], model.means_ # Retorna el estado actual y las medias de cada estado
    except:
        return np.nan, None

def compute_evt_tail_risk(series, quantile=0.95):
    """
    Calcula el riesgo de cola usando la Teoría de Valores Extremos (EVT) 
    con una distribución de Pareto Generalizada (GPD).
    Establece la probabilidad de 'eventos de cisne negro'.
    """
    try:
        # Solo nos interesan las pérdidas extremas (valores negativos)
        losses = -series[series < 0]
        threshold = losses.quantile(quantile)
        exceedances = losses[losses > threshold] - threshold
        
        # Ajustar GPD
        shape, loc, scale = genpareto.fit(exceedances)
        return {'shape': shape, 'scale': scale, 'threshold': threshold}
    except:
        return None

def check_benford_law(series):
    """
    Verifica si la distribución de los primeros dígitos sigue la Ley de Benford.
    Útil para detectar manipulación de mercado o anomalías estructurales.
    """
    try:
        first_digits = series.abs().astype(str).str[0].astype(int)
        first_digits = first_digits[first_digits > 0]
        counts = first_digits.value_counts(normalize=True).sort_index()
        
        # Probabilidades teóricas de Benford
        benford_probs = np.log10(1 + 1/np.arange(1, 10))
        
        # Test de correlación
        correlation = np.corrcoef(counts.values, benford_probs[:len(counts)])[0, 1]
        return correlation
    except:
        return np.nan

def generate_levy_flight(n_steps, alpha=1.5):
    """
    Simula un Vuelo de Lévy, que modela caminatas aleatorias de cola pesada.
    Se usa para comparar el comportamiento del mercado real con este proceso estocástico.
    """
    # Generar ángulos uniformes
    theta = np.random.uniform(0, 2*np.pi, n_steps)
    # Generar pasos según distribución de Pareto (cola pesada)
    steps = (np.random.uniform(0, 1, n_steps)**(-1/alpha))
    
    x = np.cumsum(steps * np.cos(theta))
    y = np.cumsum(steps * np.sin(theta))
    return x, y

if __name__ == "__main__":
    # Test
    returns = pd.Series(np.random.normal(0, 0.01, 200))
    
    print("Calculando Regímenes HMM...")
    state, means = compute_hmm_regimes(returns)
    print(f"Estado actual: {state}")
    
    print("\nCalculando Riesgo de Cola EVT...")
    evt = compute_evt_tail_risk(returns)
    print(evt)
    
    print("\nCorrelación con Ley de Benford...")
    print(f"Corr: {check_benford_law(returns):.4f}")
