import numpy as np

class SentimentSynth:
    """
    Generador de Sentimiento Sintético Correlacionado (Protocolo Crónica).
    Permite entrenar a los agentes en la relación entre noticias y precio
    sin necesidad de noticias históricas reales.
    """
    def __init__(self, bullish_threshold=0.01, bearish_threshold=-0.01):
        self.bull_thresh = bullish_threshold
        self.bear_thresh = bearish_threshold

    def generate_sentiment(self, future_returns):
        """
        Genera un vector de sentimiento de 3 dimensiones [Bullish, Bearish, Neutral]
        basado en el retorno futuro k-pasos adelante.
        
        Matemáticamente:
        S = softmax([P(bull), P(bear), P(neu)])
        Donde las probabilidades se sesgan según el signo de future_returns.
        """
        # Caso: Retorno Futuro Positivo (Euforia)
        if future_returns > self.bull_thresh:
            # Sesgo fuerte a Bullish: [0.90, 0.05, 0.05] + ruido
            base = np.array([0.9, 0.05, 0.05])
        
        # Caso: Retorno Futuro Negativo (Pánico)
        elif future_returns < self.bear_thresh:
            # Sesgo fuerte a Bearish: [0.05, 0.90, 0.05]
            base = np.array([0.05, 0.9, 0.05])
            
        # Caso: Lateral/Ruido (Neutral)
        else:
            # Sesgo a Neutral: [0.15, 0.15, 0.70]
            base = np.array([0.15, 0.15, 0.7])

        # Inyectar ruido gaussiano sutil (Fidelidad de Red Neuronal)
        noise = np.random.normal(0, 0.02, 3)
        raw_vector = base + noise
        
        # Asegurar que sea una distribución de probabilidad válida (Sum = 1.0)
        exp_v = np.exp(raw_vector - np.max(raw_vector))
        sentiment_vector = exp_v / exp_v.sum()
        
        return sentiment_vector.tolist()

    def generate_deceptive_sentiment(self, future_returns):
        """
        MODO 'TRAP' (Trampa de Mercado):
        Genera sentimiento CONTRARIO al movimiento futuro para entrenar la 
        resiliencia del agente y que aprenda a confiar más en el VAE (Precio) 
        cuando hay divergencias.
        """
        real_sent = self.generate_sentiment(future_returns)
        # Invertimos Bullish por Bearish
        deceptive = [real_sent[1], real_sent[0], real_sent[2]]
        return deceptive

if __name__ == "__main__":
    # Test Sanity
    synth = SentimentSynth()
    print(f"Modo Subida (+2%): {synth.generate_sentiment(0.02)}")
    print(f"Modo Caída (-2%): {synth.generate_sentiment(-0.02)}")
    print(f"Modo Lateral (0%): {synth.generate_sentiment(0.001)}")
    print(f"Modo Trampa (Subida +2%): {synth.generate_deceptive_sentiment(0.02)}")
