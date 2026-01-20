import torch
import numpy as np
import copy
from src.swarm_brain.swarm_controller import SwarmController

class EvolutionService:
    """
    Tarea 21.25: The Reaper.
    Implementa la lógica Darwiniana para el Swarm.
    Mata a los débiles. Clona a los fuertes. Muta el ADN.
    """
    def __init__(self, swarm: SwarmController, mutation_rate=0.01, cull_ratio=0.2, elite_ratio=0.1):
        self.swarm = swarm
        self.mutation_rate = mutation_rate
        self.cull_ratio = cull_ratio # Eliminar bottom 20%
        self.elite_ratio = elite_ratio # Clonar top 10%
        
    def evolve(self, metrics: dict):
        """
        Ciclo evolutivo principal.
        metrics: Dictionario {agent_id: {'pnl': float, 'sharpe': float}}
        """
        print(f"[EVO] Initiating Natural Selection on {len(self.swarm.population)} agents...")
        
        # 1. Ranking
        # Convertir metricas a lista ordenable
        ranked_agents = []
        for agent_id, data in metrics.items():
            try:
                # Parsear ID "agente_42" -> 42
                idx = int(agent_id.split('_')[1])
                score = data.get('pnl', -9999.0) # Usamos PnL como fitness primario por ahora (Tarea 21.26)
                ranked_agents.append((idx, score))
            except:
                continue
                
        # Ordenar descendente (Mayor PnL primero)
        ranked_agents.sort(key=lambda x: x[1], reverse=True)
        
        n_agents = len(ranked_agents)
        if n_agents < 10:
            print("[EVO] Population too small to evolve.")
            return

        n_cull = int(n_agents * self.cull_ratio)
        n_elite = int(n_agents * self.elite_ratio)
        
        elites = ranked_agents[:n_elite]
        culls = ranked_agents[-n_cull:]
        
        print(f"[EVO] Elites: {len(elites)} | Culls: {len(culls)}")
        
        # 2. Reemplazo (Sexual Reproduction Strategy)
        # Los peores (culls) son reemplazados por hijos de dos padres Elite distintos.
        
        # Ensure we have enough elites to breed
        if len(elites) < 2:
            print("[EVO] Not enough elites for sexual reproduction. Falling back to cloning.")
            # Fallback code or just use same parent twice
            
        for i, (cull_idx, cull_score) in enumerate(culls):
            # Select two distinct parents from elites
            parent_a_idx, _ = elites[i % len(elites)]
            parent_b_idx, _ = elites[(i + 1) % len(elites)] # Simple pairing strategy
            
            # Obtener cerebros
            brain_a = self.swarm.population[parent_a_idx]
            brain_b = self.swarm.population[parent_b_idx]
            victim_brain = self.swarm.population[cull_idx]
            
            # 3. Create Child (Crossover + Mutation)
            new_weights = self.create_child(brain_a, brain_b)
            
            # Inyectar en victima (Renacimiento)
            victim_brain.policy.load_state_dict(new_weights)
            
            print(f"[EVO] Agent_{cull_idx} (PnL {cull_score:.2f}) reborn as child of Agent_{parent_a_idx} & Agent_{parent_b_idx}.")

        print("[EVO] Evolution Complete.")

    def crossover_weights(self, state_dict_a, state_dict_b):
        """
        Tarea 21.30: Mezcla de pesos (Average Cross).
        """
        child_dict = {}
        for key in state_dict_a:
            tensor_a = state_dict_a[key]
            tensor_b = state_dict_b[key]
            # Simple average crossover
            child_dict[key] = (tensor_a + tensor_b) / 2.0
        return child_dict

    def create_child(self, parent_a, parent_b):
        """
        Tarea 21.32: Gestación completa.
        """
        weights_a = parent_a.policy.state_dict()
        weights_b = parent_b.policy.state_dict()
        
        # Crossover
        child_weights = self.crossover_weights(weights_a, weights_b)
        
        # Mutation
        child_weights = self.mutate_weights(child_weights)
        
        return child_weights

    def mutate_weights(self, state_dict):
        """
        Añade ruido gaussiano a los pesos.
        """
        for key in state_dict:
            tensor = state_dict[key]
            noise = torch.randn_like(tensor) * self.mutation_rate
            state_dict[key] = tensor + noise
        return state_dict
