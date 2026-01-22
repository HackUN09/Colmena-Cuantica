"""
COLMENA-CUÁNTICA v12.0 - State Builder
=======================================
Orquestador maestro que construye el vector de estado completo de 50 dimensiones
integrando: VAE latents + Math features + Self-awareness + Swarm intelligence

Arquitectura del Estado:
    ┌─────────────────────────────────────────────────────────┐
    │ CAPA 1: VAE Latents (27 dims)                          │
    │   - micro (8), meso (8), macro (8), sentiment (3)      │
    ├─────────────────────────────────────────────────────────┤
    │ CAPA 2: Mathematical Arsenal (16 dims)                 │
    │   - Spectral (2), Physics (4), Probability (4),        │
    │   - Statistics (2), LinAlg (2), Signals (2)            │
    ├─────────────────────────────────────────────────────────┤
    │ CAPA 3: Self-Awareness (5 dims)                        │
    │   - balance, pnl, sharpe, streak, win_rate            │
    ├─────────────────────────────────────────────────────────┤
    │ CAPA 4: Swarm Collective (3 dims)                      │
    │   - bull_ratio, avg_pnl, rank                          │
    └─────────────────────────────────────────────────────────┘
    
Total: 27 + 16 + 5 + 3 = 51 dimensiones
"""

import torch
import numpy as np
from typing import Dict, Optional
from collections import deque

from src.math_kernel.unified_feature_extractor import UnifiedFeatureExtractor
from src.math_kernel.scale_invariant import ScaleInvariantTransform
from src.math_kernel.stationarity import frac_diff_ffd


class StateBuilder:
    """
    Construye el vector de estado completo para agentes SAC.
    
    Responsabilidades:
    1. Extraer features matemáticas (UnifiedFeatureExtractor)
    2. Procesar con VAE (compresión)
    3. Añadir self-awareness metrics
    4. Añadir swarm collective intelligence
    5. Normalizar y retornar tensor PyTorch
    """
    
    def __init__(
        self,
        vae_model,
        device: str = 'cuda',
        state_dim: int = 51,
        enable_full_math: bool = True
    ):
        """
        Args:
            vae_model: Modelo VAE pre-entrenado
            device: 'cuda' o 'cpu'
            state_dim: Dimensionalidad total esperada
            enable_full_math: Si True, usa UnifiedFeatureExtractor completo
        """
        self.vae_model = vae_model
        self.device = device
        self.state_dim = state_dim
        self.enable_full_math = enable_full_math
        
        # Inicializar extractores
        if self.enable_full_math:
            self.feature_extractor = UnifiedFeatureExtractor(
                window_size=100,
                cache_heavy_computations=True
            )
        else:
            self.feature_extractor = None
            
        self.scale_transform = ScaleInvariantTransform(window_size=100)
        
        # Buffers para cálculos temporales
        self.history_buffer = deque(maxlen=100)
        
        print(f"[StateBuilder] Inicializado | State Dim: {state_dim} | Full Math: {enable_full_math}")
        
    def build_state(
        self,
        market_data: Dict,
        agent_id: str,
        treasury_manager,
        swarm_aggregator,
        current_tick: int = 0
    ) -> torch.Tensor:
        """
        Construye el estado completo de 51 dimensiones.
        
        Args:
            market_data: Dict con OHLCV de múltiples activos
            agent_id: ID del agente (para métricas individuales)
            treasury_manager: Instancia de TreasuryManager
            swarm_aggregator: Instancia de SwarmStateAggregator
            current_tick: Tick actual (para caché)
            
        Returns:
            Tensor de shape (51,) normalizado
        """
        state_components = []
        
        # CAPA 1: VAE Latents (27 dims)
        vae_latents = self._extract_vae_features(market_data)
        state_components.append(vae_latents)
        
        # CAPA 2: Mathematical Arsenal (16 dims)
        if self.enable_full_math:
            math_features = self._extract_math_features(market_data, current_tick)
            state_components.append(math_features)
        else:
            # Fallback: zeros
            state_components.append(np.zeros(16))
            
        # CAPA 3: Self-Awareness (5 dims)
        self_features = self._extract_self_features(agent_id, treasury_manager)
        state_components.append(self_features)
        
        # CAPA 4: Swarm Collective (3 dims)
        swarm_features = self._extract_swarm_features(agent_id, swarm_aggregator)
        state_components.append(swarm_features)
        
        # Concatenar todo
        state_vector = np.concatenate(state_components)
        
        # Verificar dimensionalidad
        assert len(state_vector) == self.state_dim, \
            f"Expected {self.state_dim} dims, got {len(state_vector)}"
            
        # Handle NaNs
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # Convertir a tensor
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        
        return state_tensor
        
    def _extract_vae_features(self, market_data: Dict) -> np.ndarray:
        """
        Extrae latents de VAE (micro, meso, macro) + sentiment.
        
        Returns:
            Array de 27 dims: [z_micro(8), z_meso(8), z_macro(8), z_sentiment(3)]
        """
        # TODO: Implementar extraction completa con VAE
        # Por ahora retornar placeholder
        
        # Simular VAE latents (temporal)
        z_micro = np.random.randn(8) * 0.1
        z_meso = np.random.randn(8) * 0.1
        z_macro = np.random.randn(8) * 0.1
        z_sentiment = np.array([0.33, 0.33, 0.34])  # Neutral
        
        vae_features = np.concatenate([z_micro, z_meso, z_macro, z_sentiment])
        
        return vae_features
        
    def _extract_math_features(self, market_data: Dict, tick: int) -> np.ndarray:
        """
        Extrae features matemáticas usando UnifiedFeatureExtractor.
        
        Returns:
            Array de 16 dims: [spectral(2), physics(4), prob(4), stats(2), linalg(2), signals(2)]
        """
        if self.feature_extractor is None:
            return np.zeros(16)
            
        try:
            features_dict = self.feature_extractor.extract(market_data, tick)
            
            # Orden fijo de features
            feature_order = [
                'dominant_period', 'spectral_entropy',  # Spectral (2)
                'hurst_exponent', 'fractal_dimension', 'shannon_entropy', 'lyapunov_exponent',  # Physics (4)
                'regime_bear', 'regime_neutral', 'regime_bull', 'evt_tail_risk',  # Probability (4)
                'garch_volatility', 'arima_forecast_error',  # Statistics (2)
                'rmt_signal_strength', 'pca_variance_explained',  # LinAlg (2)
                'wavelet_noise_ratio', 'kalman_smoothness',  # Signals (2)
            ]
            
            feature_vector = []
            for key in feature_order:
                value = features_dict.get(key, 0.0)
                feature_vector.append(value)
                
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"[StateBuilder] Error extracting math features: {e}")
            return np.zeros(16)
            
    def _extract_self_features(self, agent_id: str, treasury) -> np.ndarray:
        """
        Extrae métricas de self-awareness del agente.
        
        Returns:
            Array de 5 dims: [balance_norm, pnl_mean, sharpe, loss_streak, win_rate]
        """
        try:
            # Balance normalizado
            balance = treasury.agentes_ledger.get(agent_id, 1000.0)
            initial_capital = 1000.0  # Hardcoded por ahora
            balance_norm = balance / initial_capital
            
            # PnL reciente (últimos 10 ticks)
            pnl_history = treasury.agent_history.get(agent_id, deque(maxlen=10))
            if len(pnl_history) > 0:
                pnls = [x[2] for x in pnl_history]  # (timestamp, balance, pnl)
                pnl_mean = np.mean(pnls)
                pnl_std = np.std(pnls) + 1e-10
                sharpe_10t = pnl_mean / pnl_std
            else:
                pnl_mean = 0.0
                sharpe_10t = 0.0
                
            # Loss streak
            loss_streak = treasury.loss_streaks.get(agent_id, 0)
            
            # Win rate (últimos 20 trades)
            if len(pnl_history) >= 20:
                recent_20 = list(pnl_history)[-20:]
                wins = sum(1 for x in recent_20 if x[2] > 0)
                win_rate = wins / 20
            else:
                win_rate = 0.5  # Neutral
                
            return np.array([
                balance_norm,
                pnl_mean,
                sharpe_10t,
                float(loss_streak) / 10.0,  # Normalizar
                win_rate
            ], dtype=np.float32)
            
        except Exception as e:
            print(f"[StateBuilder] Error extracting self features: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
            
    def _extract_swarm_features(self, agent_id: str, swarm_agg) -> np.ndarray:
        """
        Extrae métricas del swarm collective.
        
        Returns:
            Array de 3 dims: [bull_ratio, avg_pnl, rank]
        """
        try:
            # Consensus (ratio de agentes alcistas)
            bull_ratio, bear_ratio, neutral_ratio = swarm_agg.get_consensus()
            
            # PnL promedio del swarm
            avg_pnl = swarm_agg.get_avg_pnl()
            
            # Ranking del agente
            rank = swarm_agg.get_agent_rank(agent_id)
            
            return np.array([
                bull_ratio,
                avg_pnl,
                rank
            ], dtype=np.float32)
            
        except Exception as e:
            print(f"[StateBuilder] Error extracting swarm features: {e}")
            return np.array([0.33, 0.0, 0.5], dtype=np.float32)


if __name__ == "__main__":
    print("=" * 70)
    print("Test: StateBuilder")
    print("=" * 70)
    
    # Mock VAE
    class MockVAE:
        def __init__(self):
            pass
            
    vae = MockVAE()
    
    # Mock Treasury
    class MockTreasury:
        def __init__(self):
            self.agentes_ledger = {"agent_0": 1050.0}
            self.agent_history = {"agent_0": deque([(0, 1000, 10), (1, 1010, 5)], maxlen=10)}
            self.loss_streaks = {"agent_0": 2}
            
    treasury = MockTreasury()
    
    # Mock Swarm
    class MockSwarm:
        def get_consensus(self):
            return 0.4, 0.3, 0.3  # bull, bear, neutral
        def get_avg_pnl(self):
            return 15.0
        def get_agent_rank(self, agent_id):
            return 0.75  # Top 25%
            
    swarm = MockSwarm()
    
    # Mock market data
    market_data = {
        'BTC/USDT': [[1000000, 50000, 50100, 49900, 50000, 1000]],
        'ETH/USDT': [[1000000, 3000, 3050, 2950, 3000, 500]]
    }
    
    # Create builder
    builder = StateBuilder(vae, device='cpu', enable_full_math=True)
    
    # Build state
    print("\n[TEST] Building state vector...")
    state = builder.build_state(market_data, "agent_0", treasury, swarm, tick=0)
    
    print(f"\n[RESULT]")
    print(f"  Shape: {state.shape}")
    print(f"  Device: {state.device}")
    print(f"  Dtype: {state.dtype}")
    print(f"  Min: {state.min():.4f}, Max: {state.max():.4f}")
    print(f"  Mean: {state.mean():.4f}, Std: {state.std():.4f}")
    print(f"  NaNs: {torch.isnan(state).sum().item()}")
    
    print("\n✅ StateBuilder test completed")
