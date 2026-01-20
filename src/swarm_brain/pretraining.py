import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from src.swarm_brain.agent_sac import SoftActorCritic
from src.vae_layer.nlp_bert import NLPSentimentEngine

class Pretrainer:
    def __init__(self, symbols=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = SoftActorCritic()
        self.nlp = NLPSentimentEngine(service_url="http://localhost:8001") # Acceso desde host
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']

    def descargar_datos_reales(self, limit=500):
        """Descarga datos históricos reales de Binance."""
        all_data = {}
        print(f"[PRE] Descargando ráfaga técnica de Binance ({limit} velas)...")
        for symbol in self.symbols:
            try:
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit={limit}"
                res = requests.get(url).json()
                # Timestamp, Open, High, Low, Close
                data = [[int(c[0]), float(c[4])] for c in res]
                all_data[symbol] = data
                print(f"  > {symbol}: OK")
            except Exception as e:
                print(f"  > Error en {symbol}: {e}")
        return all_data

    def sintetizar_sentimiento_historico(self, returns_vec):
        """
        Crea un sentimiento sintético correlacionado con la acción del precio.
        Esto enseña al agente la RELACIÓN CAUSA-EFECTO:
        - Grandes caídas -> Pánico (Negativo)
        - Grandes subidas -> Euforia (Positivo)
        - Lateralización -> Neutralidad
        """
        avg_return = np.mean(returns_vec)
        volatility = np.std(returns_vec)
        
        # Vector: [Positivo, Negativo, Neutral]
        if avg_return < -0.005: # Caída > 0.5% en una hora
            return [0.1, 0.8, 0.1] # Pánico
        elif avg_return > 0.005: # Subida > 0.5% en una hora
            return [0.8, 0.1, 0.1] # Euforia
        else:
            return [0.1, 0.1, 0.8] # Neutral

    def ejecutar_estudio_historico(self, epochs=50):
        # 1. Obtener datos técnicos (Binance)
        raw_data = self.descargar_datos_reales()
        if not raw_data: return
        
        # 2. Sentimiento Base del Presente (como sesgo contextual)
        current_mood = self.obtener_noticias_actuales()
        
        # Sincronizar timestamps
        df_prices = pd.DataFrame({s: [x[1] for x in raw_data[s]] for s in self.symbols})
        returns = df_prices.pct_change().dropna()
        
        print(f"[PRE] Iniciando Entrenamiento de Causalidad (Precio <-> Narrativa)...")
        print(f"[PRE] Estudiando {len(returns)} horas de historia real.")

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(returns)):
                current_returns = returns.iloc[i].values
                
                # REGLA DE ORO: Usamos sentimiento sintético CORRELACIONADO con esa hora del pasado
                # Esto es lo que enseña al agente a REACCIONAR al miedo/euforia.
                historical_sentiment = self.sintetizar_sentimiento_historico(current_returns)
                
                state_vec = np.zeros(11)
                state_vec[:4] = current_returns
                state_vec[8:] = historical_sentiment # Aquí está la causalidad real
                
                state_t = torch.FloatTensor(state_vec).to(self.device).view(1, -1)
                
                # Política toma acción
                action = self.agent.policy.sample(state_t)
                action_np = action.cpu().detach().numpy()[0]
                
                # Recompensa: PnL real en esa ventana histórica
                reward = np.dot(action_np[:4], current_returns)
                
                # Optimización
                loss = self.agent.update(state_t, action_np, reward)
                total_loss += loss

            if epoch % 10 == 0:
                print(f"[PRE] Epoch {epoch} | Loss: {total_loss/len(returns):.6f}")

        # Guardar Cerebro Graduado
        if not os.path.exists("models"): os.makedirs("models")
        torch.save(self.agent.policy.state_dict(), "models/cerebro_base_colmena.pth")
        print("\n[ÉXITO] Los 100 agentes han sido graduados con datos históricos reales.")

if __name__ == "__main__":
    pt = Pretrainer()
    pt.ejecutar_estudio_historico()
