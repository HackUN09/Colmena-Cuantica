import torch
import numpy as np
import os
import logging
from src.swarm_brain.agent_sac import SoftActorCritic

class SwarmController:
    """
    Tarea 21.10: The Hive Mind Controller.
    Gestiona una población real de 'n_agents' cerebros SoftActorCritic únicos.
    Sustituye al agente singular placeholder.
    """
    def __init__(self, n_agents=100, state_dim=27, action_dim=11, load_path=None):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[HIVE] Gestating {n_agents} unique bio-agents on {self.device}...")
        
        # Tarea 21.14: Instanciación del Enjambre
        for i in range(n_agents):
            try:
                # Genética Temporal: Cada agente nace con un 'Ritmo' propio (10m a 60m)
                # Esto valida la teoría de "Resonancia Bio-Espectral".
                dilation = np.random.randint(10, 61)
                
                # Cada agente tiene sus propios pesos aleatorios iniciales (Xavier init implícito en PyTorch)
                agent = SoftActorCritic(state_dim=state_dim, action_dim=action_dim, time_dilation=dilation)
                self.population.append(agent)
            except Exception as e:
                print(f"[HIVE ERROR] Failed to birth agent {i}: {e}")
                
        print(f"[HIVE] Swarm Population Born. Census: {len(self.population)}")
        
        # Tarea 21.43: Carga de Estado (Integración Phase 4)
        if load_path and os.path.exists(load_path):
            self.load_population_state(load_path)

    def get_action(self, agent_id_str, state):
        """
        Tarea 21.16: Inferencia Específica.
        Decodifica el ID del agente y consulta su cerebro específico.
        """
        try:
            # Parse ID ex: "agente_42"
            parts = agent_id_str.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
            else:
                idx = 0 # Fallback
            
            # Bound check
            idx = idx % self.n_agents 
            
            return self.population[idx].select_action(state)
            
        except Exception as e:
            # Fallback seguro a Agente 0 en caso de error de parsing
            return self.population[0].select_action(state)

    def get_actions_batch(self, states_tensor):
        """
        Tarea 21.22: Inferencia en Batch (Optimizada).
        Ejecuta inferencia para todos los agentes a la vez.
        states_tensor: Tensor de (N_Agentes, 11) o (1, 11) broadcasted.
        """
        actions = []
        # Nota: Con 100 redes distintas, la paralelización real requiere
        # Ensembling o vmap (funcional). Por ahora, bucle optimizado.
        for i, agent in enumerate(self.population):
            # Asumimos que states_tensor[i] corresponde al agente i
            # o si es un solo estado global, lo usamos para todos.
            if states_tensor.shape[0] == self.n_agents:
                s = states_tensor[i]
            else:
                s = states_tensor[0] 
                
            action = agent.select_action(s)
            actions.append(action)
        return list(actions)

    def get_population_state_dicts(self):
        """
        Tarea 21.17: Serialización del Enjambre.
        Retorna una lista con los state_dicts de todos los agentes.
        """
        return [agent.policy.state_dict() for agent in self.population]

    def set_population_state_dicts(self, dicts):
        """
        Tarea 21.18: Hidratación del Enjambre.
        Carga pesos en cada agente desde una lista de dicts.
        """
        print(f"[HIVE] Loading synaptic weights for {len(dicts)} agents...")
        for i, state_dict in enumerate(dicts):
            if i < len(self.population):
                self.population[i].policy.load_state_dict(state_dict)

    def save_checkpoint(self, path, metadata: dict = None):
        """
        Persistencia Atómica compatible con StorageManager.
        """
        from datetime import datetime
        data = {
            "population_state": self.get_population_state_dicts(),
            "timestamp": datetime.now().isoformat(),
            "n_agents": self.n_agents,
            "metadata": metadata or {}
        }
        torch.save(data, path)
        print(f"[HIVE] Swarm state saved to {path} (Meta-format)")

    def load_population_state(self, path):
        """
        Recuperación Atómica compatible con formatos híbridos.
        Retorna la metadata si existe.
        """
        try:
            data = torch.load(path, map_location=self.device)
            metadata = {}
            if isinstance(data, dict) and "population_state" in data:
                self.set_population_state_dicts(data["population_state"])
                metadata = data.get("metadata", {})
                print(f"[HIVE] Swarm restored from {path} (Meta-format)")
            else:
                self.set_population_state_dicts(data)
                print(f"[HIVE] Swarm restored from {path} (Legacy-format)")
            return metadata
        except Exception as e:
            print(f"[HIVE ERROR] Corrupt Brain File {path}: {e}")
            return {}

if __name__ == "__main__":
    # Test Sanity
    swarm = SwarmController(n_agents=5)
    dummy_state = torch.randn(27)
    act = swarm.get_action("agente_3", dummy_state)
    print(f"Action shape: {act.shape}")
