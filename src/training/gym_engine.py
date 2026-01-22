import torch
import random
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.vae_layer.model import VAE
from src.training.sentiment_synth import SentimentSynth
from src.swarm_brain.swarm_controller import SwarmController
from src.swarm_brain.genetics import EvolutionService
from src.execution.treasury_manager import TreasuryManager
from src.math_kernel.stationarity import frac_diff_ffd
from src.math_kernel.ticker_universe import TICKER_UNIVERSE, get_ticker_index

class GymEngine:
    """
    EL GIMNASIO DE ALTA FIDELIDAD (Protocolo Fourier v10.0).
    Orquestador de entrenamiento fractal sobre Top 10 Crypto Assets.
    Orquestador de entrenamiento fractal sobre Top 10 Crypto Assets.
    Implementa 'Resonancia Bio-Espectral' construyendo tensores temporales on-the-fly.
    """
    def __init__(self, n_agents=100, storage_dir='data/historical', seed=42):
        # 0. Reproducibilidad Científica
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.swarm = SwarmController(n_agents=n_agents)
        
        # 1. Validación Dimensional
        assert self.swarm.state_dim == 27, f"Error Fractal: Tensor esperado 27d, recibido {self.swarm.state_dim}d"
        assert self.swarm.action_dim == 11, f"Error Topología: Acción esperada 11d (10 Assets + Cash)"
        
        self.vae = VAE(input_dim=10, latent_dim=8).to(self.device) # Top 10 -> 8 Latent
        self.vae.eval()
        self.synth = SentimentSynth()
        self.treasury = TreasuryManager(master_reserve_addr="USDT")
        self.evolution = EvolutionService(self.swarm, cull_ratio=0.2, elite_ratio=0.1)
        self.storage_dir = storage_dir
        
    def load_portfolio(self, timeframe='1m'):
        """
        Carga y alinea los Top 10 activos.
        """
        # Buscar archivos solo del Universo Top 10 para ahorrar memoria
        target_files = []
        for ticker in TICKER_UNIVERSE:
            safe_ticker = ticker.replace('/', '_')
            fname = f"{safe_ticker}_{timeframe}.csv"
            if os.path.exists(os.path.join(self.storage_dir, fname)):
                target_files.append(fname)
        
        if not target_files:
            print("[GYM ERROR] No hay datos del Universo Top 10 para cargar.")
            return None, None
            
        print(f"[GYM] Cargando portafolio Fourier ({len(target_files)} activos)...")
        price_data = {}
        ffd_data = {}
        
        for f in target_files:
            ticker = f.split(f"_{timeframe}")[0].replace('_', '/')
            df = pd.read_csv(os.path.join(self.storage_dir, f))
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Guardar precios y FFD
            price_data[ticker] = df['close']
            df_closes = df[['close']].copy()
            # Diferenciación Fraccionaria Real
            res_ffd = frac_diff_ffd(df_closes, d=0.4).fillna(0.0)
            ffd_data[ticker] = res_ffd.iloc[:, 0]

        # Alinear DataFrames
        valid_tickers = [t for t in TICKER_UNIVERSE if t in price_data]
        
        df_prices = pd.DataFrame({t: price_data[t] for t in valid_tickers}).sort_index().ffill().dropna()
        df_ffd = pd.DataFrame({t: ffd_data[t] for t in valid_tickers}).sort_index().ffill().dropna()
        
        common_index = df_prices.index.intersection(df_ffd.index)
        return df_prices.loc[common_index], df_ffd.loc[common_index]

    def _encode_layer(self, vector):
        """Helper para pasar un vector por el VAE y obtener espacio latente (8 dims)"""
        tensor = torch.as_tensor(vector, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            _, mu, _ = self.vae(tensor)
        return mu.flatten()

    def train_portfolio(self, iterations=1, k_future=15):
        """
        Bucle de Entrenamiento Fractal.
        """
        df_prices, df_ffd = self.load_portfolio()
        if df_prices is None: return
        
        tickers = list(df_ffd.columns)
        print(f"[GYM] Iniciando entrenamiento Bio-Espectral sobre {len(tickers)} activos...")
        
        self.treasury.capitalizar_colmena(1000.0, self.swarm.n_agents)
        
        # --- RESILIENCIA (SOPORTE RESUME) ---
        checkpoint_path = 'models/cerebro_checkpoint.pth'
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            print(f"[RECOVER] Detectado checkpoint anterior. Sincronizando...")
            try:
                metadata = self.swarm.load_population_state(checkpoint_path)
                start_epoch = metadata.get("last_epoch", 0)
                if start_epoch > 0:
                    print(f"[RECOVER] Resumiendo desde Generación {start_epoch}...")
            except Exception as e:
                print(f"[RECOVER WARNING] Error cargando checkpoint: {e}. Iniciando Fresh.")
        
        # Si ya se alcanzó la meta, preguntamos
        if start_epoch >= iterations:
            print(f"[GYM] La meta de {iterations} generaciones ya ha sido alcanzada anteriormente ({start_epoch}).")
            return

        # MACRO WINDOW (Tendencia): 4 horas = 240 minutos
        MACRO_WINDOW = 240
        
        for epoch in range(start_epoch, iterations):
            # Empezamos en MACRO_WINDOW para tener historial suficiente
            start_idx = MACRO_WINDOW + 60 
            pbar = tqdm(range(start_idx, len(df_prices) - k_future), desc=f"Gen {epoch+1}", mininterval=1.0, dynamic_ncols=True, leave=True)
            
            for t in pbar:
                # 1. CAPA MICRO (1m): La realidad inmediata
                micro_vec = df_ffd.iloc[t].values # Shape (10,)
                
                # 2. CAPA MACRO (4h): El Clima
                # Promedio del FFD en las últimas 4 horas (intensidad de tendencia sostenida)
                macro_vec = df_ffd.iloc[t-MACRO_WINDOW:t].mean().values # Shape (10,)
                
                # Pre-codificar capas globales para eficiencia
                z_micro = self._encode_layer(micro_vec)
                z_macro = self._encode_layer(macro_vec)
                
                # Sentimiento Global
                prices_now = df_prices.iloc[t].values
                prices_future = df_prices.iloc[t + k_future].values
                retorno_indice = np.mean((prices_future / prices_now) - 1)
                sentiment = self.synth.generate_sentiment(retorno_indice)
                z_sentiment = torch.as_tensor(sentiment, dtype=torch.float32, device=self.device)
                
                # Retornos para calcular PnL
                prices_prev = df_prices.iloc[t-1].values
                retornos_paso = (prices_now / prices_prev) - 1
                
                # BUCLE DE AGENTES (Individualidad Fractal)
                for i, agent in enumerate(self.swarm.population):
                    # 3. CAPA MESO (Genética): La visión única del agente
                    dilation = agent.time_dilation 
                    # Promedio en su ventana de dilación
                    meso_vec = df_ffd.iloc[t-dilation:t].mean().values
                    z_meso = self._encode_layer(meso_vec)
                    
                    # CONSTRUCCIÓN DEL TENSOR FRACTAL (27 dims)
                    # [Micro (8) + Meso (8) + Macro (8) + Sentiment (3)]
                    state = torch.cat([z_micro, z_meso, z_macro, z_sentiment], dim=0)
                    
                    agente_id = f"agente_{i}"
                    action = self.swarm.get_action(agente_id, state)
                    balance = self.treasury.agentes_ledger[agente_id]
                    
                    # Calcular PnL y Volumen para aplicación de comisiones
                    pnl_total = 0.0
                    vol_agente = 0.0
                    for idx_asset, ticker in enumerate(tickers):
                        weight = action[idx_asset]
                        ret = retornos_paso[idx_asset]
                        pnl_total += ret * balance * weight
                        vol_agente += abs(weight) * balance
                        
                    # 4. Aprendizaje (Con Fricción Realista)
                    agent.update(state, action, pnl_total)
                    self.treasury.procesar_cierre_orden(agente_id, pnl_total, volume_usd=vol_agente)
                    
                if t % 100 == 0:
                     resumen = self.treasury.obtener_resumen_enjambre()
                     pbar.set_postfix({"Cap": f"${resumen['total_capital_activo']:.2f}"})

            # --- EVOLUCIÓN ---
            print("\n[GYM] SELECCIÓN NATURAL (Protocolo Darwin)...")
            metrics = {id: {'pnl': b - 10.0} for id, b in self.treasury.agentes_ledger.items()}
            self.evolution.evolve(metrics)
            
            self.treasury.capitalizar_colmena(1000.0, self.swarm.n_agents)
            self.save_brain('models/cerebro_checkpoint.pth', metadata={"last_epoch": epoch + 1})
            print(f"[GYM] Generación {epoch+1} completada.\n")

    def save_brain(self, path='models/cerebro_colmena_entrenado.pth', metadata: dict = None):
        # Atomic Save: Write to .tmp first, then rename.
        tmp_path = path + ".tmp"
        try:
            self.swarm.save_checkpoint(tmp_path, metadata=metadata)
            if os.path.exists(tmp_path):
                os.replace(tmp_path, path) # Atomic operation on POSIX/Windows (mostly)
                print(f"[GYM] Brain saved successfully (Atomic): {path}")
        except Exception as e:
            print(f"[SAVE ERROR] Critical failure saving brain: {e}")

if __name__ == "__main__":
    gym = GymEngine()
    try:
        # Test de Carga
        prices, ffd = gym.load_portfolio()
        if prices is not None:
            print(f"[TEST] Tensor Shape: {prices.shape}")
            # gym.train_portfolio(iterations=1)
    except Exception as e:
        print(f"[TEST ERROR] {e}")
