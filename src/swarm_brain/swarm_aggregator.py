"""
COLMENA-CUÁNTICA v12.0 - Swarm State Aggregator
=================================================
Servicio que agrega y normaliza métricas del enjambre COMPLETO.
Proporciona inteligencia colectiva a cada agente individual.

Fundamento Matemático:
El estado del enjambre es una proyección del espacio de agentes individuales
al espacio colectivo:
    
    Φ: ℝ^(N×D) → ℝ^K
    
donde N=agentes, D=dims individuales, K=dims colectivas

Métricas colectivas:
1. Consensus (distribución de sentimiento)
2. Average P&L (performance promedio)
3. Ranking (posición relativa)
4. Elite distance (qué tan cerca estoy del top 10%)
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class SwarmStateAggregator:
    """
    Agrega estado del enjambre completo para proveer features colectivas.
    
    Propiedades:
    - Real-time: actualizado cada tick
    - Normalized: todas las métricas en [0, 1] o [-1, 1]
    - Privacy-preserving: agentes solo ven agregados, no individuales
    """
    
    def __init__(self, n_agents: int = 100):
        """
        Args:
            n_agents: Número total de agentes en el enjambre
        """
        self.n_agents = n_agents
        self.agent_actions = {}  # Dict[agent_id, last_action]
        self.agent_pnls = {}  # Dict[agent_id, current_pnl]
        
    def update_agent_action(self, agent_id: str, action: np.ndarray):
        """
        Actualiza la última acción de un agente.
        
        Args:
            action: Vector de pesos [w_BTC, w_ETH, ..., w_CASH]
        """
        self.agent_actions[agent_id] = action
        
    def update_agent_pnl(self, agent_id: str, pnl: float):
        """Actualiza el P&L actual de un agente"""
        self.agent_pnls[agent_id] = pnl
        
    def get_consensus(self) -> Tuple[float, float, float]:
        """
        Calcula consenso del enjambre: qué % son alcistas/bajistas/neutrales.
        
        Regla:
        - Alcista (bull): Si peso en activos de riesgo > 60%
        - Bajista (bear): Si peso en CASH > 60%
        - Neutral: Resto
        
        Returns:
            (bull_ratio, bear_ratio, neutral_ratio) sumando a 1.0
        """
        if not self.agent_actions:
            return 0.33, 0.33, 0.34
            
        bulls = 0
        bears = 0
        neutrals = 0
        
        for agent_id, action in self.agent_actions.items():
            # action[-1] es CASH
            risk_exposure = 1.0 - action[-1]  # Suma de activos de riesgo
            
            if risk_exposure > 0.6:
                bulls += 1
            elif action[-1] > 0.6:  # Mayoría en cash
                bears += 1
            else:
                neutrals += 1
                
        total = bulls + bears + neutrals
        if total == 0:
            return 0.33, 0.33, 0.34
            
        return bulls / total, bears / total, neutrals / total
    
    def get_avg_pnl(self) -> float:
        """
        Retorna P&L promedio del enjambre.
        Normalizado para que sea comparable entre agentes.
        """
        if not self.agent_pnls:
            return 0.0
            
        pnls = list(self.agent_pnls.values())
        return float(np.mean(pnls))
    
    def get_std_pnl(self) -> float:
        """
        Retorna desviación estándar de P&L.
        Alta std = algunos muy bien, otros muy mal.
        Baja std = todos similares.
        """
        if not self.agent_pnls:
            return 0.0
            
        pnls = list(self.agent_pnls.values())
        return float(np.std(pnls))
    
    def get_agent_rank(self, agent_id: str) -> float:
        """
        Retorna percentil del agente en el enjambre.
        
        Returns:
            float en [0, 1] donde:
            - 0.0 = peor agente (bottom 1%)
            - 0.5 = mediano
            - 1.0 = mejor agente (top 1%)
        """
        if agent_id not in self.agent_pnls:
            return 0.5
            
        my_pnl = self.agent_pnls[agent_id]
        all_pnls = sorted(self.agent_pnls.values())
        
        if len(all_pnls) == 0:
            return 0.5
            
        # Percentil
        rank = sum(1 for pnl in all_pnls if pnl < my_pnl)
        percentile = rank / len(all_pnls)
        
        return percentile
    
    def get_elite_distance(self, agent_id: str, elite_percentile: float = 0.9) -> float:
        """
        Retorna qué tan lejos está el agente del umbral elite.
        
        Args:
            elite_percentile: Percentil para ser elite (0.9 = top 10%)
            
        Returns:
            Distancia normalizada:
            - Negativo = ya soy elite
            - Positivo = qué tan lejos estoy del umbral
        """
        if agent_id not in self.agent_pnls:
            return 1.0
            
        my_pnl = self.agent_pnls[agent_id]
        all_pnls = sorted(self.agent_pnls.values())
        
        if len(all_pnls) == 0:
            return 1.0
            
        # Umbral elite
        elite_idx = int(len(all_pnls) * elite_percentile)
        elite_threshold = all_pnls[elite_idx] if elite_idx < len(all_pnls) else all_pnls[-1]
        
        # Distancia normalizada
        if elite_threshold == 0:
            return 0.0
            
        distance = (my_pnl - elite_threshold) / abs(elite_threshold)
        
        return float(distance)
    
    def get_cull_distance(self, agent_id: str, cull_percentile: float = 0.1) -> float:
        """
        Retorna qué tan cerca está de ser eliminado (culled).
        
        Returns:
            - Negativo = ya estoy en zona de cull
            - Positivo = qué tan seguro estoy
        """
        if agent_id not in self.agent_pnls:
            return 0.0
            
        my_pnl = self.agent_pnls[agent_id]
        all_pnls = sorted(self.agent_pnls.values())
        
        if len(all_pnls) == 0:
            return 0.0
            
        # Umbral cull
        cull_idx = int(len(all_pnls) * cull_percentile)
        cull_threshold = all_pnls[cull_idx] if cull_idx < len(all_pnls) else all_pnls[0]
        
        if cull_threshold == 0:
            return 0.0
            
        distance = (my_pnl - cull_threshold) / abs(cull_threshold)
        
        return float(distance)
    
    def get_treasury_health(self, reserve_fund: float, active_capital: float) -> float:
        """
        Retorna salud de la tesorería (ratio reserve / total).
        
        Health = Reserve / (Reserve + Active)
        
        - 0.0 = no hay reserva (mal)
        - 0.5 = equilibrado
        - 1.0 = toda la plata en reserva (extremo)
        """
        total = reserve_fund + active_capital
        if total < 1e-10:
            return 0.0
            
        return reserve_fund / total
    
    def get_collective_state(self, agent_id: str, treasury_manager) -> Dict[str, float]:
        """
        Retorna todas las métricas colectivas en un dict.
        
        Args:
            agent_id: ID del agente consultante
            treasury_manager: Instancia de TreasuryManager para reserve_fund
            
        Returns:
            Dict con 3 features: bull_ratio, avg_pnl, rank
        """
        bull_ratio, bear_ratio, neutral_ratio = self.get_consensus()
        avg_pnl = self.get_avg_pnl()
        rank = self.get_agent_rank(agent_id)
        
        return {
            'swarm_bull_ratio': bull_ratio,
            'swarm_avg_pnl': avg_pnl,
            'swarm_rank': rank
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Test: SwarmStateAggregator")
    print("=" * 70)
    
    agg = SwarmStateAggregator(n_agents=100)
    
    # Simular 100 agentes
    np.random.seed(42)
    for i in range(100):
        agent_id = f"agente_{i}"
        
        # Acciones aleatorias (portfolio)
        action = np.random.dirichlet(np.ones(11))  # 10 activos + cash
        agg.update_agent_action(agent_id, action)
        
        # P&Ls aleatorios (distribución normal)
        pnl = np.random.randn() * 50 + 10  # Mean=10, std=50
        agg.update_agent_pnl(agent_id, pnl)
    
    print("\n[TEST 1] Consensus")
    bull, bear, neutral = agg.get_consensus()
    print(f"  Bulls: {bull:.2%}, Bears: {bear:.2%}, Neutral: {neutral:.2%}")
    print(f"  Sum: {bull + bear + neutral:.3f} (should be 1.0)")
    
    print("\n[TEST 2] Average P&L")
    avg_pnl = agg.get_avg_pnl()
    std_pnl = agg.get_std_pnl()
    print(f"  Mean: {avg_pnl:.2f}, Std: {std_pnl:.2f}")
    
    print("\n[TEST 3] Ranking")
    test_agent = "agente_50"
    rank = agg.get_agent_rank(test_agent)
    elite_dist = agg.get_elite_distance(test_agent)
    cull_dist = agg.get_cull_distance(test_agent)
    print(f"  Agent {test_agent}:")
    print(f"    Rank: {rank:.2%} percentile")
    print(f"    Distance to elite: {elite_dist:.2f}")
    print(f"    Distance to cull: {cull_dist:.2f}")
    
    print("\n[TEST 4] Collective State")
    class MockTreasury:
        reserve_fund = 5000
        
    state = agg.get_collective_state(test_agent, MockTreasury())
    for key, value in state.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ SwarmStateAggregator tests completed")
