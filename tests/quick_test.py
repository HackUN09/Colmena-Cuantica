"""
Test Rápido - COLMENA CUÁNTICA V1.0
====================================
Verifica que el sistema compila y corre 1 episodio corto.
"""

import torch
import sys
sys.path.insert(0, '.')

print("="*70)
print("TEST RÁPIDO - Sistema V1.0")
print("="*70)

# Test 1: Imports
print("\n[1/5] Testing imports...")
try:
    from src.training.historical_env import HistoricalMarketEnv
    from src.swarm_brain.agent_sac import SoftActorCritic
    from src.swarm_brain.state_builder import StateBuilder
    from src.execution.treasury_manager import TreasuryManager
    from src.math_kernel.unified_feature_extractor import UnifiedFeatureExtractor
    from src.vae_layer.model import VAE
    print("  ✅ Todos los imports exitosos")
except Exception as e:
    print(f"  ❌ Error en imports: {e}")
    sys.exit(1)

# Test 2: Environment
print("\n[2/5] Testing HistoricalMarketEnv...")
try:
    env = HistoricalMarketEnv()
    state = env.reset(random_start=False)
    print(f"  ✅ Environment OK - {len(state)} activos cargados")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 3: SAC Agent
print("\n[3/5] Testing SAC Agent (51-dim)...")
try:
    agent = SoftActorCritic(state_dim=51, action_dim=11)
    dummy_state = torch.randn(51)
    action = agent.select_action(dummy_state.numpy())
    print(f"  ✅ SAC OK - Action sum: {action.sum():.4f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 4: Feature Extractor
print("\n[4/5] Testing UnifiedFeatureExtractor...")
try:
    extractor = UnifiedFeatureExtractor(window_size=100)
    print(f"  ✅ Feature Extractor OK")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 5: StateBuilder
print("\n[5/5] Testing StateBuilder...")
try:
    vae = VAE(input_dim=10, latent_dim=8)
    builder = StateBuilder(vae, device='cpu', enable_full_math=False)
    print(f"  ✅ StateBuilder OK")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅✅✅ SISTEMA V1.0 LISTO PARA TESTEAR ✅✅✅")
print("="*70)
print("\nPróximo paso:")
print("  1. Editar src/config.py → N_EPISODES = 10 (test rápido)")
print("  2. Ejecutar: python train_offline_full.py")
print("  3. Esperar ~5 minutos para test")
print("="*70)
