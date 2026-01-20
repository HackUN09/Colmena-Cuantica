import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from src.swarm_brain.swarm_controller import SwarmController
from src.execution.storage_manager import StorageManager

class RealWorldGraduate:
    """
    Protocolo de Graduación de Agentes:
    Descarga historia real de Binance y entrena al enjambre antes del despliegue vivo.
    """
    def __init__(self, tickers=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tickers = tickers or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
        self.swarm = SwarmController(n_agents=100)
        self.storage = StorageManager()

    def fetch_history(self, symbol, limit=100):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}"
        try:
            res = requests.get(url).json()
            # [Time, Open, High, Low, Close, Vol]
            return [float(c[4]) for c in res]
        except:
            return []

    def graduate_swarm(self, epochs=20):
        print(f"[GRADUATE] Iniciando pre-entrenamiento con {len(self.tickers)} activos reales...")
        market_data = {t: self.fetch_history(t) for t in self.tickers}
        
        for epoch in range(epochs):
            total_reward = 0
            for i in range(1, 100): # Usamos ventana de 100 minutos
                # Sintetizar estado (retornos + sentimiento base)
                rets = []
                for t in self.tickers:
                    prices = market_data[t]
                    if len(prices) > i:
                        rets.append((prices[i] / prices[i-1]) - 1)
                    else:
                        rets.append(0.0)
                
                # Mock state: 11 dims (8 market + 3 sentiment)
                state = torch.zeros(1, 11).to(self.device)
                state[0, :min(len(rets), 8)] = torch.tensor(rets[:8])
                state[0, 8:] = torch.tensor([0.2, 0.1, 0.7]) # Neutral bias
                
                # Inferencia en batch para los 100 agentes
                for agent_idx, agent in enumerate(self.swarm.population):
                    action = agent.select_action(state)
                    # Reward: dot product of action and returns
                    reward = np.dot(action[:len(rets)], rets)
                    agent.update(state.cpu().numpy(), action, reward)
                    total_reward += reward
            
            print(f"  > Epoch {epoch+1}/{epochs} | Recompensa Acumulada: {total_reward:.4f}")

        self.storage.save_swarm_state(self.swarm)
        print("[ÉXITO] Enjambre graduado y guardado en Akasha. Listos para el n8n.")

if __name__ == "__main__":
    grad = RealWorldGraduate()
    grad.graduate_swarm()
