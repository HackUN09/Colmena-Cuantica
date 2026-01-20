import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpectralAnalysis")

class SpectralAnalyzer:
    """
    Núcleo Matemático del Protocolo Fourier v10.0.
    Responsabilidad: Escuchar la frecuencia natural del mercado (Bio-Resonancia).
    """

    def __init__(self, sampling_rate_min=1):
        """
        Args:
            sampling_rate_min: Resolución base (1 minuto).
        """
        self.fs = 1.0 / sampling_rate_min  # Frecuencia de muestreo (1/min)

    def compute_psd(self, log_returns):
        """
        Calcula la Densidad Espectral de Potencia (PSD) usando el método de Welch.
        Reduce la varianza del estimador dividiendo la señal en segmentos solapados.
        """
        # Usamos nperseg razonable para capturar bajas frecuencias pero tener promedios
        nperseg = min(len(log_returns), 256) 
        freqs, psd = welch(log_returns, fs=self.fs, nperseg=nperseg)
        return freqs[1:], psd[1:] # Descartar componente DC (0 Hz)

    def fit_red_noise(self, freqs, psd):
        """
        Ajusta un modelo de Ruido Rojo (1/f^alpha) al espectro de fondo.
        Log(PSD) ~ -alpha * Log(f) + C
        """
        # Regresión lineal en escala Log-Log
        log_f = np.log10(freqs)
        log_p = np.log10(psd)
        
        # Ajuste simple: y = mx + c
        # Alpha suele estar entre 1 (Pink) y 2 (Brownian)
        coeffs = np.polyfit(log_f, log_p, 1)
        slope, intercept = coeffs
        
        # Modelo teórico
        red_noise_psd = 10**(slope * log_f + intercept)
        return red_noise_psd, -slope # Retornamos alpha positivo

    def calculate_spectral_entropy(self, psd):
        """
        Calcula la Entropía de Shannon del Espectro Normalizado (SE).
        SE bajo -> Señal tonal/rítmica (Predecible).
        SE alto -> Ruido blanco/caos (Impredecible).
        """
        psd_norm = psd / np.sum(psd)
        se = entropy(psd_norm, base=2)
        return se

    def find_habitable_zones(self, freqs, psd, alpha_threshold=0.05):
        """
        Identifica bandas de frecuencia donde la señal supera significativamente al ruido rojo.
        Retorna: Lista de tuplas (Periodo_Min_Minutos, Periodo_Max_Minutos)
        """
        red_noise, alpha = self.fit_red_noise(freqs, psd)
        
        # Intervalo de Confianza 95% (Chi-Cuadrado aproximado 2*PSD_teorico)
        # Simplificación: Buscamos donde PSD_real > 2 * PSD_ruido
        confidence_bound = 2.0 * red_noise
        
        habitable_indices = np.where(psd > confidence_bound)[0]
        
        zones = []
        if len(habitable_indices) == 0:
            return zones
            
        # Agrupar índices contiguos
        # (Lógica simplificada para prototipo v1)
        current_zone = [habitable_indices[0]]
        
        for i in range(1, len(habitable_indices)):
            if habitable_indices[i] == habitable_indices[i-1] + 1:
                current_zone.append(habitable_indices[i])
            else:
                # Cerrar zona anterior
                # Convertir freqs a Periodo (T = 1/f)
                f_start = freqs[current_zone[0]]
                f_end = freqs[current_zone[-1]]
                # T_min corresponde a f_end (frecuencia alta)
                t_min = 1.0 / f_end
                t_max = 1.0 / f_start
                zones.append((t_min, t_max))
                current_zone = [habitable_indices[i]]
                
        # Última zona
        if current_zone:
            f_start = freqs[current_zone[0]]
            f_end = freqs[current_zone[-1]]
            t_min = 1.0 / f_end
            t_max = 1.0 / f_start
            zones.append((t_min, t_max))
            
        return zones

if __name__ == "__main__":
    # Test Unitario con onda sinusoidal ruidosa
    analyzer = SpectralAnalyzer()
    t = np.linspace(0, 1000, 1000)
    # Señal: Onda de 43 min + Ruido
    signal = np.sin(2 * np.pi * t / 43.0) + 0.5 * np.random.randn(1000)
    
    freqs, psd = analyzer.compute_psd(signal)
    zones = analyzer.find_habitable_zones(freqs, psd)
    se = analyzer.calculate_spectral_entropy(psd)
    
    print(f"Spectral Entropy: {se:.4f}")
    print(f"Habitable Zones (Period mins): {zones}")
