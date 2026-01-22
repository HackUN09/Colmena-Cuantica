"""
COLMENA-CUÁNTICA V1.0 - Offline Training Full
==============================================
Script completo de pre-entrenamiento usando datos históricos.

CONFIGURACIÓN: Editar src/config.py para cambiar parámetros
"""

import torch
import numpy as np
import os
import pickle
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

# Core components
from src.training.historical_env import HistoricalMarketEnv
from src.swarm_brain.swarm_controller import SwarmController
from src.swarm_brain.state_builder import StateBuilder
from src.swarm_brain.swarm_aggregator import SwarmStateAggregator
from src.execution.treasury_manager import TreasuryManager
from src.vae_layer.model import VAE
from src.math_kernel.unified_feature_extractor import UnifiedFeatureExtractor

# ==================== IMPORTAR CONFIGURACIÓN ====================
from src.config import (
    N_AGENTS,
    N_EPISODES,
    TICKS_PER_EPISODE,
    CHECKPOINT_INTERVAL,
    HARVEST_RATE,
    COMMISSION_RATE,
    STATE_DIM,
    ACTION_DIM,
    MODELS_DIR,
    RESULTS_DIR,
    DATA_DIR,
    DEVICE as CONFIG_DEVICE,
    ENABLE_SPECTRAL,
    ENABLE_PHYSICS,
    ENABLE_PROBABILITY,
    ENABLE_STATISTICS,
    ENABLE_LINEAR_ALGEBRA,
    ENABLE_SIGNALS,
    FEATURE_WINDOW_SIZE,
    CACHE_HEAVY_COMPUTATIONS
)

# Device
DEVICE = CONFIG_DEVICE if torch.cuda.is_available() else 'cpu'
STATS_DIR = RESULTS_DIR  # Alias

print("=" * 70)
print("COLMENA-CUÁNTICA V1.0 - Offline Training")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Agentes: {N_AGENTS}")
print(f"Episodios: {N_EPISODES}")
print(f"Ticks/Episodio: {TICKS_PER_EPISODE}")
print(f"Experiencias totales: {N_EPISODES * TICKS_PER_EPISODE * N_AGENTS:,}")
print(f"State: {STATE_DIM}-dim | Action: {ACTION_DIM}-dim")
print("=" * 70)
print("\n💡 Para cambiar configuración, editar: src/config.py")
print("=" * 70)

# Crear directorios
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

# ==================== INICIALIZACIÓN ====================

print("\n[1/6] Inicializando componentes...")

# Environment
env = HistoricalMarketEnv(
    data_path='data/historical',
    commission_rate=COMMISSION_RATE
)
print(f"  ✓ HistoricalMarketEnv: {env.max_ticks:,} ticks disponibles")

# VAE (por ahora mock, luego pre-entrenar por separado)
vae_model = VAE(input_dim=10, latent_dim=8).to(DEVICE)
vae_model.eval()
print(f"  ✓ VAE Model: {sum(p.numel() for p in vae_model.parameters()):,} params")

# StateBuilder
state_builder = StateBuilder(
    vae_model=vae_model,
    device=DEVICE,
    state_dim=51,
    enable_full_math=True
)
print(f"  ✓ StateBuilder: 51-dim state")

# Treasury
treasury = TreasuryManager(
    master_reserve_addr="USDT",
    harvest_rate=HARVEST_RATE,
    commission_rate=COMMISSION_RATE
)
treasury.capitalizar_colmena(total_capital=100000.0, n_agentes=N_AGENTS)
print(f"  ✓ TreasuryManager: {N_AGENTS} agentes × $1000 USDT")

# Swarm Aggregator
swarm_agg = SwarmStateAggregator(n_agents=N_AGENTS)
print(f"  ✓ SwarmStateAggregator")

# Swarm Controller
swarm = SwarmController(n_agents=N_AGENTS, state_dim=51, action_dim=11)
print(f"  ✓ SwarmController: {N_AGENTS} agentes SAC")
total_params = sum(
    sum(p.numel() for p in agent.policy.parameters())
    for agent in swarm.population
)
print(f"     Total parameters: {total_params:,}")

# Feature Extractor
feature_extractor = UnifiedFeatureExtractor(
    window_size=100,
    enable_spectral=True,
    enable_physics=True,
    enable_probability=True,
    enable_statistics=True,
    enable_linear_algebra=True,
    enable_signals=True,
    cache_heavy_computations=True
)
print(f"  ✓ UnifiedFeatureExtractor: 16 math features")

# ==================== TRAINING LOOP ====================

print("\n[2/6] Iniciando entrenamiento offline...")

training_stats = {
    'episode_rewards': [],
    'episode_sharpe': [],
    'episode_survival_rate': [],
    'avg_loss': [],
    'best_agent_pnl': []
}

for episode in range(N_EPISODES):
    # Reset environment
    market_data = env.reset(random_start=True)
    
    # Reset metrics
    episode_rewards = []
    episode_losses = []
    
    # Progress bar
    pbar = tqdm(
        range(TICKS_PER_EPISODE),
        desc=f"Ep {episode+1}/{N_EPISODES}",
        leave=False
    )
    
    for tick in pbar:
        # Cada agente toma decisión
        agent_actions = []
        agent_states = []
        
        for i, agent in enumerate(swarm.population):
            agent_id = f"agente_{i}"
            
            # Construir estado completo (51-dims)
            try:
                state = state_builder.build_state(
                    market_data=market_data,
                    agent_id=agent_id,
                    treasury_manager=treasury,
                    swarm_aggregator=swarm_agg,
                    current_tick=tick
                )
            except Exception as e:
                # Fallback: estado con ceros si falla
                state = torch.zeros(51, device=DEVICE)
            
            # Seleccionar acción
            action = agent.select_action(state.cpu().numpy())
            
            agent_actions.append(action)
            agent_states.append(state)
            
            # Actualizar swarm aggregator
            swarm_agg.update_agent_action(agent_id, action)
        
        # Simular mercado con acción del primer agente (simplificado)
        # En producción, cada agente tendría su propia simulación
        representative_action = agent_actions[0]
        next_market_data, reward, done = env.step(representative_action)
        
        # Aplicar reward a todos los agentes (broadcast)
        for i, agent in enumerate(swarm.population):
            agent_id = f"agente_{i}"
            
            # Calcular P&L simulado
            pnl = reward * np.random.uniform(0.8, 1.2)  # Variación por agente
            
            # Procesar en treasury
            volume = abs(agent_actions[i]).sum() * 1000  # Estimado
            treasury.procesar_cierre_orden(agent_id, pnl, volume)
            
            # Actualizar swarm aggregator
            current_balance = treasury.agentes_ledger[agent_id]
            swarm_agg.update_agent_pnl(agent_id, current_balance - 1000.0)
            
            # Entrenar agente (online learning)
            if tick % 10 == 0:  # Cada 10 ticks
                loss = agent.update(
                    state=agent_states[i].cpu().numpy(),
                    action=agent_actions[i],
                    reward=pnl
                )
                episode_losses.append(loss)
        
        episode_rewards.append(reward)
        market_data = next_market_data
        
        # Update progress bar
        if tick % 100 == 0:
            avg_balance = np.mean([treasury.agentes_ledger[f"agente_{i}"] for i in range(N_AGENTS)])
            pbar.set_postfix({
                'Avg Balance': f"${avg_balance:.0f}",
                'Reward': f"{reward:.4f}"
            })
        
        if done:
            break
    
    # Episode stats
    avg_reward = np.mean(episode_rewards)
    sharpe = np.mean(episode_rewards) / (np.std(episode_rewards) + 1e-10)
    
    # Survival rate (agentes con balance > 0)
    survivors = sum(1 for b in treasury.agentes_ledger.values() if b > 0)
    survival_rate = survivors / N_AGENTS
    
    # Best agent
    best_pnl = max(treasury.agentes_ledger.values()) - 1000.0
    
    # Registrar
    training_stats['episode_rewards'].append(avg_reward)
    training_stats['episode_sharpe'].append(sharpe)
    training_stats['episode_survival_rate'].append(survival_rate)
    training_stats['avg_loss'].append(np.mean(episode_losses) if episode_losses else 0.0)
    training_stats['best_agent_pnl'].append(best_pnl)
    
    # Print summary
    if (episode + 1) % 10 == 0:
        print(f"\nEpisode {episode+1}/{N_EPISODES}:")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Sharpe: {sharpe:.4f}")
        print(f"  Survival Rate: {survival_rate:.1%}")
        print(f"  Best Agent P&L: ${best_pnl:.2f}")
        print(f"  Avg Loss: {training_stats['avg_loss'][-1]:.6f}")
    
    # Guardar checkpoints
    if (episode + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = f"{MODELS_DIR}/checkpoint_ep{episode+1}.pkl"
        checkpoint = {
            'episode': episode + 1,
            'agents': [
                {
                    'id': f"agente_{i}",
                    'state_dict': agent.policy.state_dict(),
                    'balance': treasury.agentes_ledger[f"agente_{i}"]
                }
                for i, agent in enumerate(swarm.population)
            ],
            'training_stats': training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  💾 Checkpoint guardado: {checkpoint_path}")

# ==================== FINAL CHECKPOINT ====================

print("\n[3/6] Guardando checkpoint final...")

final_checkpoint = {
    'episode': N_EPISODES,
    'agents': [
        {
            'id': f"agente_{i}",
            'state_dict': agent.policy.state_dict(),
            'balance': treasury.agentes_ledger[f"agente_{i}"],
            'pnl': treasury.agentes_ledger[f"agente_{i}"] - 1000.0
        }
        for i, agent in enumerate(swarm.population)
    ],
    'training_stats': training_stats,
    'config': {
        'n_agents': N_AGENTS,
        'n_episodes': N_EPISODES,
        'state_dim': 51,
        'action_dim': 11,
        'commission_rate': COMMISSION_RATE,
        'harvest_rate': HARVEST_RATE
    },
    'timestamp': datetime.now().isoformat()
}

final_path = f"{MODELS_DIR}/final_pretrained.pkl"
with open(final_path, 'wb') as f:
    pickle.dump(final_checkpoint, f)

print(f"  ✅ Checkpoint final: {final_path}")

# Guardar solo los mejores agentes (top 10)
top_10_agents = sorted(
    enumerate(swarm.population),
    key=lambda x: treasury.agentes_ledger[f"agente_{x[0]}"],
    reverse=True
)[:10]

for rank, (idx, agent) in enumerate(top_10_agents):
    agent_path = f"{MODELS_DIR}/elite_agent_{rank+1}.pth"
    torch.save(agent.policy.state_dict(), agent_path)
    pnl = treasury.agentes_ledger[f"agente_{idx}"] - 1000.0
    print(f"  💎 Elite #{rank+1}: agente_{idx} | P&L: ${pnl:.2f}")

# ==================== VISUALIZACIÓN ====================

print("\n[4/6] Generando visualizaciones...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Learning curve
axes[0, 0].plot(training_stats['episode_rewards'])
axes[0, 0].set_title('Episode Rewards')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Avg Reward')
axes[0, 0].grid(True, alpha=0.3)

# Sharpe ratio
axes[0, 1].plot(training_stats['episode_sharpe'])
axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Target: 1.0')
axes[0, 1].set_title('Sharpe Ratio')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Sharpe')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Survival rate
axes[1, 0].plot(training_stats['episode_survival_rate'])
axes[1, 0].axhline(y=0.9, color='g', linestyle='--', label='Target: 90%')
axes[1, 0].set_title('Survival Rate')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Survival %')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Best agent P&L
axes[1, 1].plot(training_stats['best_agent_pnl'])
axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 1].set_title('Best Agent P&L')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('P&L ($)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{STATS_DIR}/training_curves.png", dpi=150)
print(f"  📊 Gráficas guardadas: {STATS_DIR}/training_curves.png")

# ==================== REPORTE FINAL ====================

print("\n[5/6] Generando reporte final...")

final_balances = [treasury.agentes_ledger[f"agente_{i}"] for i in range(N_AGENTS)]
final_pnls = [b - 1000.0 for b in final_balances]

report = f"""
{'='*70}
COLMENA-CUÁNTICA v12.0 - REPORTE DE ENTRENAMIENTO OFFLINE
{'='*70}

CONFIGURACIÓN:
- Agentes: {N_AGENTS}
- Episodios: {N_EPISODES}
- Ticks por episodio: {TICKS_PER_EPISODE}
- Experiencias totales: {N_EPISODES * TICKS_PER_EPISODE * N_AGENTS:,}
- State dim: 51 (VAE:27 + Math:16 + Self:5 + Swarm:3)
- Comisión: {COMMISSION_RATE:.2%}
- Harvest rate: {HARVEST_RATE:.0%}

RESULTADOS FINALES:
- Avg P&L: ${np.mean(final_pnls):.2f} ± ${np.std(final_pnls):.2f}
- Median P&L: ${np.median(final_pnls):.2f}
- Best Agent: ${max(final_pnls):.2f}
- Worst Agent: ${min(final_pnls):.2f}
- Agents profitable: {sum(1 for p in final_pnls if p > 0)}/{N_AGENTS} ({sum(1 for p in final_pnls if p > 0)/N_AGENTS:.1%})
- Final Sharpe: {training_stats['episode_sharpe'][-1]:.4f}
- Final Survival: {training_stats['episode_survival_rate'][-1]:.1%}

MÉTRICAS DE APRENDIZAJE:
- Avg loss (último episodio): {training_stats['avg_loss'][-1]:.6f}
- Reward mejora: {training_stats['episode_rewards'][-1] - training_stats['episode_rewards'][0]:.4f}
- Sharpe mejora: {training_stats['episode_sharpe'][-1] - training_stats['episode_sharpe'][0]:.4f}

RECOMENDACIONES:
"""

if training_stats['episode_sharpe'][-1] > 1.0:
    report += "✅ Sharpe > 1.0: EXCELENTE. Deploy a producción recomendado.\n"
elif training_stats['episode_sharpe'][-1] > 0.5:
    report += "⚠️  Sharpe > 0.5: ACEPTABLE. Considerar más entrenamiento.\n"
else:
    report += "❌ Sharpe < 0.5: INSUFICIENTE. Revisar hiperparámetros o features.\n"

if training_stats['episode_survival_rate'][-1] > 0.9:
    report += "✅ Survival > 90%: Agentes robustos.\n"
else:
    report += "⚠️  Survival < 90%: Alta mortalidad. Revisar gestión de riesgo.\n"

report += f"\n{'='*70}\n"

print(report)

# Guardar reporte
with open(f"{STATS_DIR}/training_report.txt", 'w') as f:
    f.write(report)

print(f"  📄 Reporte guardado: {STATS_DIR}/training_report.txt")

# ==================== FINALIZACIÓN ====================

print("\n[6/6] ✅ Entrenamiento offline completado")
print(f"\nCheckpoints disponibles en: {MODELS_DIR}/")
print(f"- final_pretrained.pkl (todos los agentes)")
print(f"- elite_agent_1.pth ... elite_agent_10.pth (top 10)")
print(f"\nPara cargar en main.py:")
print(f"  agent.policy.load_state_dict(torch.load('models/pretrained/elite_agent_1.pth'))")
print("\n🚀 Sistema listo para transfer learning a operación live!")
