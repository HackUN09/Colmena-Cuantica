import numpy as np
import pandas as pd
from scipy.fft import fft
import pywt
from pykalman import KalmanFilter

def apply_fft_analysis(series):
    """
    Transformada Rápida de Fourier para detectar periodicidades dominantes (ciclos).
    """
    try:
        y = series.dropna().values
        n = len(y)
        yf = fft(y)
        xf = np.linspace(0.0, 1.0, n//2)
        # Espectro de potencia
        psd = 2.0/n * np.abs(yf[0:n//2])
        return xf[np.argmax(psd)], np.max(psd) # Frecuencia dominante y su potencia
    except:
        return np.nan, np.nan

def apply_wavelet_denoising(series, wavelet='db4', level=1):
    """
    Transformada Wavelet Discreta para suavizar la serie sin perder los picos críticos.
    Ideal para series financieras no estacionarias.
    """
    try:
        coeffs = pywt.wavedec(series.dropna(), wavelet, mode='per', level=level)
        # Umbralización suave (Soft thresholding)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(series)))
        coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
        
        reconstructed = pywt.waverec(coeffs, wavelet, mode='per')
        return pd.Series(reconstructed[:len(series)], index=series.index)
    except:
        return series

def apply_kalman_filter(series):
    """
    Filtro de Kalman para estimar el 'estado real' oculto del precio.
    Actúa como un promedio móvil inteligente adaptativo.
    """
    try:
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=series.iloc[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)
        state_means, _ = kf.filter(series.values)
        return pd.Series(state_means.flatten(), index=series.index)
    except:
        return series

def apply_particle_filter_estimation(series, n_particles=100):
    """
    Aproximación de Filtro de Partículas para estados no lineales/no gaussianos.
    """
    try:
        particles = np.random.normal(series.iloc[0], 1, n_particles)
        weights = np.ones(n_particles) / n_particles
        results = []
        
        for val in series.values:
            # Predicción (Random Walk)
            particles += np.random.normal(0, 0.1, n_particles)
            # Actualización (Likelihood)
            weights *= np.exp(-0.5 * (particles - val)**2)
            weights += 1e-12 # Evitar división por cero
            weights /= np.sum(weights)
            
            # Estimación
            results.append(np.sum(particles * weights))
            
            # Resampling simplificado
            if 1.0 / np.sum(weights**2) < n_particles / 2:
                idx = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
                particles = particles[idx]
                weights = np.ones(n_particles) / n_particles
                
        return pd.Series(results, index=series.index)
    except:
        return series

if __name__ == "__main__":
    # Test
    data = pd.Series(np.sin(np.linspace(0, 5, 100)) + np.random.normal(0, 0.2, 100))
    print("Aplicando Kalman...")
    klm = apply_kalman_filter(data)
    print(f"Kalman suavizado (último valor): {klm.iloc[-1]:.4f}")
    
    print("\nAplicando Wavelet Denoising...")
    wav = apply_wavelet_denoising(data)
    print(f"Wavelet suavizado (último valor): {wav.iloc[-1]:.4f}")
