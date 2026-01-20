import torch
import uvicorn
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from typing import List, Dict, Optional, Deque
from datetime import datetime
import collections

# --- Importaciones Quirúrgicas del Ecosistema COLMENA ---
from src.math_kernel.stationarity import frac_diff_ffd
from src.vae_layer.model import VAE
from src.vae_layer.nlp_bert import NLPSentimentEngine
from src.swarm_brain.agent_sac import SoftActorCritic
from src.execution.risk_manager import RiskManager
from src.execution.treasury_manager import TreasuryManager
import asyncio
import random
import math
import psycopg2
from src.math_kernel.ticker_universe import TICKER_UNIVERSE, get_ticker_index

# --- Inicialización Maestra ---
app = FastAPI(title="CORE COLMENA-CUÁNTICA // PROTOCOLO FOURIER (V10.0)")

# Detección de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[BOOT] Iniciando núcleos en: {device}")

# Carga de Motores en VRAM
vae_model = VAE(input_dim=10, latent_dim=8).to(device) # Input 10 (Top 10 Assets)
vae_model.eval() 
nlp_engine = NLPSentimentEngine()

# --- Deep Truth Core Modules ---
from src.swarm_brain.swarm_controller import SwarmController
from src.swarm_brain.genetics import EvolutionService
from src.execution.storage_manager import StorageManager
from src.swarm_brain.replay_buffer import SharedReplayBuffer

# --- Initialize Subsystems ---
print("[BOOT] Initializing Bio-Spectral Swarm...")
swarm = SwarmController(n_agents=100) # Instancia agentes con time_dilation aleatorio
storage_manager = StorageManager()
replay_buffer = SharedReplayBuffer(capacity=50000)

# MARKET MEMORY: Buffer Fractal (Micro-history)
# Necesitamos al menos 240 mins para la Capa Macro
MARKET_MEMORY: Deque[np.ndarray] = collections.deque(maxlen=300)

# PROTOCOLO GÉNESIS
if storage_manager.load_swarm_state(swarm):
    print("[BOOT] Swarm Memory Restored from Akasha.")
else:
    print("[BOOT] Genesis Protocol: Swarm initialized with fresh bio-agents.")

print("[BOOT] Initializing Evolution Service (The Reaper)...")
evolution = EvolutionService(swarm, cull_ratio=0.2, elite_ratio=0.1)

# Risk & Treasury
risk_engine = RiskManager()
treasury = TreasuryManager(master_reserve_addr="USDT", harvest_rate=0.20)

# --- Funciones DB ---
def sync_treasury_to_db(ledger, reserve):
    try:
        conn = psycopg2.connect(host="postgres", database="n8n", user="n8n_user", password="n8n_pass", port="5432")
        with conn.cursor() as cur:
            for ag_id, balance in ledger.items():
                cur.execute("INSERT INTO wallets (agente_id, balance) VALUES (%s, %s) ON CONFLICT (agente_id) DO UPDATE SET balance = EXCLUDED.balance;", (ag_id, balance))
            cur.execute("INSERT INTO transactions (timestamp, agente_id, pnl_bruto, tax_cosecha, balance_resultante) VALUES (%s, %s, %s, %s, %s);", (datetime.now(), "SINDICATO_COLMENA", 0.0, reserve, sum(ledger.values())))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB SYNC ERROR] {e}")

def load_treasury_from_db(treasury_instance):
    try:
        conn = psycopg2.connect(host="postgres", database="n8n", user="n8n_user", password="n8n_pass", port="5432")
        with conn.cursor() as cur:
            cur.execute("SELECT agente_id, balance FROM wallets;")
            rows = cur.fetchall()
            if rows:
                print(f"[Treasury] Restaurando {len(rows)} carteras desde DB...")
                for ag_id, balance in rows:
                    treasury_instance.agentes_ledger[ag_id] = float(balance)
                return True
        conn.close()
    except Exception as e:
        print(f"[DB LOAD ERROR] {e}")
    return False

if load_treasury_from_db(treasury):
    print("[BOOT] Tesorería restaurada.")
else:
    print("[BOOT] DB vacía. Hard Reset a $1000.00.")
    treasury.capitalizar_colmena(total_capital=1000.0, n_agentes=100)
    sync_treasury_to_db(treasury.agentes_ledger, treasury.reserve_fund)

# --- Estados Globales ---
last_states = {} 
last_actions = {} 
study_pulse = 0
total_operations = 0

async def colmena_heartbeat():
    global study_pulse
    if not hasattr(swarm, 'real_data_hit'): swarm.real_data_hit = False
    
    while True:
        try:
            if swarm.real_data_hit:
                study_pulse += 1
                swarm.real_data_hit = False
            
            if study_pulse >= 60:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] PROTOCOLO GÉNESIS: Retraining...")
                if len(replay_buffer) > 100:
                    for _ in range(3): 
                        batch = replay_buffer.sample(min(len(replay_buffer), 128))
                        for agent in swarm.population:
                            agent.update(batch[0][0].cpu().numpy(), batch[1][0].cpu().numpy(), batch[2][0].item())
                study_pulse = 0
            
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[REAPER ERROR] {e}")
            await asyncio.sleep(5)

# --- Estructura de Datos ---
class IngestData(BaseModel):
    ticker_data: Dict[str, List]
    news_headlines: List[str]
    balances: Optional[Dict[str, float]] = None

class InputValidator:
    @staticmethod
    def validate(data: IngestData) -> bool:
        return bool(data.ticker_data)

def extract_market_features(data: IngestData) -> np.ndarray:
    """Extrae vector FFD de 10 dimensiones (Top 10) usando TICKER_UNIVERSE."""
    features = np.zeros(10) # 10 Assets
    
    for ticker, ohlcv in data.ticker_data.items():
        # idx es 0..9 para Top 10
        idx = get_ticker_index(ticker)
        if 0 <= idx < 10:
            if len(ohlcv) < 20: continue
            try:
                closes = [float(c[4]) for c in ohlcv]
                df_series = pd.DataFrame(closes, columns=['close'])
                diffed = frac_diff_ffd(df_series, d=0.4).fillna(0.0)
                if not diffed.empty:
                    val = diffed.iloc[-1].values[0]
                    # Normalización Tanh-like
                    norm_val = np.tanh(val) # -1 a 1
                    features[idx] = norm_val
            except:
                continue
    return features

# --- Endpoints Operativos ---
@app.get("/")
async def root():
    return {"sistema": "COLMENA-CUÁNTICA v10.0 (Bio-Spectral)", "status": "ONLINE"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(colmena_heartbeat())

@app.post("/procesar_inferencia")
async def procesar_inferencia(data: IngestData):
    global total_operations, study_pulse
    if not InputValidator.validate(data): return {"status": "error"}

    # 1. Ingesta y Memoria Fractal
    current_features = extract_market_features(data) # (10,)
    MARKET_MEMORY.append(current_features)
    
    # Necesitamos buffers numpy para cálculos vectorizados
    buffer_array = np.array(MARKET_MEMORY) # (N, 10)
    
    # 2. Sentimiento
    sentiment_vec = nlp_engine.get_sentiment_vector(data.news_headlines)
    z_sentiment = torch.tensor(sentiment_vec, dtype=torch.float32, device=device)
    
    # 3. Pre-Cálculo de Capas Globales (Micro y Macro)
    # Micro: Último tick
    micro_vec = current_features
    # Macro: Últimos 240 ticks (o lo que haya)
    macro_lookback = min(len(MARKET_MEMORY), 240)
    macro_vec = buffer_array[-macro_lookback:].mean(axis=0)
    
    # Codificar Globales
    with torch.no_grad():
        _, z_micro, _ = vae_model(torch.tensor([micro_vec], dtype=torch.float32, device=device))
        _, z_macro, _ = vae_model(torch.tensor([macro_vec], dtype=torch.float32, device=device))
        z_micro = z_micro.flatten()
        z_macro = z_macro.flatten()

    # 4. Inferencia por Agente (Capa Meso Fractal)
    results = {}
    current_actions = []
    
    # Cache para optimizar dilaciones repetidas
    meso_cache = {} 
    
    # Si tenemos balances frescos de n8n
    if data.balances:
        for ag_id, bal in data.balances.items():
            treasury.agentes_ledger[ag_id] = bal

    agentes_list = list(treasury.agentes_ledger.keys())
    
    for i, agente_id in enumerate(agentes_list):
        try:
            agent = swarm.population[i]
            dilation = agent.time_dilation
            
            # Calcular Meso State (Lazy Cache)
            if dilation not in meso_cache:
                lookback = min(len(MARKET_MEMORY), dilation)
                meso_vec = buffer_array[-lookback:].mean(axis=0)
                with torch.no_grad():
                     _, z_meso, _ = vae_model(torch.tensor([meso_vec], dtype=torch.float32, device=device))
                meso_cache[dilation] = z_meso.flatten()
            
            z_meso = meso_cache[dilation]
            
            # Construir Tensor Fractal: [Micro(8) + Meso(8) + Macro(8) + Sentiment(3)] = 27 dims
            state = torch.cat([z_micro, z_meso, z_macro, z_sentiment], dim=0)
            
            # INFERENCIA
            action = swarm.get_action(agente_id, state)
            current_actions.append("COMPRA" if action[0] > 0.5 else "VENTA")
            
            # PnL Calculation (Top 10)
            total_real_pnl = 0.0
            balance = treasury.agentes_ledger[agente_id]
            
            # Calcular retornos
            for ticker, ohlcv in data.ticker_data.items():
                idx = get_ticker_index(ticker)
                if 0 <= idx < 10 and len(ohlcv) >= 2:
                    p_now = float(ohlcv[-1][4])
                    p_prev = float(ohlcv[-2][4])
                    ret = (p_now / p_prev) - 1
                    # Peso del activo en el portfolio del agente
                    w = action[idx]
                    pnl = ret * balance * w
                    total_real_pnl += pnl
                    
            # Aprendizaje
            if agente_id in last_states:
                replay_buffer.push(last_states[agente_id], last_actions[agente_id], total_real_pnl, state, False)
            
            last_states[agente_id] = state
            last_actions[agente_id] = action
            
            # Ledger e Informe
            harvest = treasury.procesar_cierre_orden(agente_id, total_real_pnl)
            results[agente_id] = {"pnl_total": total_real_pnl, "auditoria": harvest["auditoria_tesoro"]}

            # AUDIT LOG (Solo para Agente 0 para no saturar)
            if i == 0:
                 print(f"[AUDIT] Agente 0 ({dilation}m) -> Acciones: {action[:5].round(2)}... PnL: ${total_real_pnl:.6f}")

        except Exception as e:
            continue

    swarm.real_data_hit = True
    sync_treasury_to_db(treasury.agentes_ledger, treasury.reserve_fund)
    
    return {
        "status": "success", 
        "study_pulse": study_pulse, 
        "resumen": treasury.obtener_resumen_enjambre(),
        "inferencias": results
    }

# --- Monitor de Salud (Health Check) ---
@app.get("/health")
def health_check():
    """
    Endpoint de diagnóstico para balanceadores de carga y monitoring.
    """
    return {
        "status": "operational",
        "pulse": "V1.0 Bio-Spectral",
        "gpu_available": torch.cuda.is_available(),
        "agents_active": len(swarm.population),
        "treasury_balance": sum(treasury.agentes_ledger.values())
    }

# --- Emergency Reset (Solo Admin) ---
@app.post("/reset_system")
def reset_system(password: str):
    if password != "PROTOCOL_OMEGA": return {"error": "Unauthorized"}
    global MARKET_MEMORY
    MARKET_MEMORY.clear()
    return {"status": "System memory purged. Fresh start initiated."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
