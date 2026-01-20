import torch
import os
import shutil
from datetime import datetime
import logging

class StorageManager:
    """
    Tarea 21.37: Persistence Manager (The Akasha).
    Maneja el ciclo de vida de los archivos del cerebro del enjambre,
    incluyendo guardado, carga, backups y manejo de 'Génesis'.
    """
    def __init__(self, base_path="models", brain_filename="cerebro_colmena.pth"):
        self.base_path = base_path
        self.brain_path = os.path.join(base_path, brain_filename)
        self.backup_path = os.path.join(base_path, "backups")
        
        # Asegurar existencia de directorios
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_swarm_state(self, swarm_controller, transaction_id=None):
        """
        Tarea 21.38: Persiste el estado completo del enjambre.
        """
        try:
            # Obtener dicts de la población y del controlador en sí si fuera necesario
            population_state = swarm_controller.get_population_state_dicts()
            
            meta_data = {
                "timestamp": datetime.now().isoformat(),
                "n_agents": swarm_controller.n_agents,
                "transaction_id": transaction_id,
                "population_state": population_state
            }
            
            # Guardado Atómico (Write temp -> Move)
            temp_path = self.brain_path + ".tmp"
            torch.save(meta_data, temp_path)
            shutil.move(temp_path, self.brain_path)
            
            print(f"[STORAGE] Swarm saved successfully to {self.brain_path}")
            return True
        except Exception as e:
            print(f"[STORAGE ERROR] Failed to save swarm: {e}")
            return False

    def load_swarm_state(self, swarm_controller):
        """
        Tarea 21.40: Carga el estado del enjambre.
        Maneja el caso GENESIS (file not found) retornando False.
        """
        if not os.path.exists(self.brain_path):
            print(f"[STORAGE] No brain found at {self.brain_path}. Initiating GENESIS protocol.")
            return False
            
        try:
            checkpoint = torch.load(self.brain_path, map_location=self.device)
            
            # Validación básica
            if "population_state" not in checkpoint:
                raise ValueError("Checkpoint corrupted: missing 'population_state'")
                
            population_state = checkpoint["population_state"]
            swarm_controller.set_population_state_dicts(population_state)
            
            ts = checkpoint.get("timestamp", "Unknown")
            print(f"[STORAGE] Swarm restored (Timestamp: {ts})")
            return True
            
        except Exception as e:
            print(f"[STORAGE ERROR] Corrupt Brain File: {e}")
            # Intentar restaurar backup si existiera (fase futura)
            return False

    def create_snapshot(self):
        """
        Tarea 21.45: Backup periódico.
        """
        if os.path.exists(self.brain_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"swarm_backup_{timestamp}.pth"
            dest = os.path.join(self.backup_path, backup_name)
            shutil.copy2(self.brain_path, dest)
            print(f"[STORAGE] Snapshot created: {backup_name}")
            return dest
        return None
