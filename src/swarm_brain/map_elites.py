import numpy as np

class MAPElites:
    """
    Implementación de Calidad-Diversidad vía MAP-Elites.
    Mantiene un archivo (grid) de los mejores agentes según su comportamiento.
    """
    def __init__(self, dims=(20, 20)): # Grid 20x20
        # Ejes: X = Volatilidad Aguantada, Y = Rotación de Cartera
        self.grid_dims = dims
        self.archive = {} # Key: (idx_x, idx_y), Value: { 'agent_weights': ..., 'performance': ... }

    def _get_grid_index(self, volatility, turnover):
        """
        Mapea el comportamiento a una celda del grid.
        Assume inputs normalizados [0, 1].
        """
        idx_x = int(volatility * (self.grid_dims[0] - 1))
        idx_y = int(turnover * (self.grid_dims[1] - 1))
        return (idx_x, idx_y)

    def update_archive(self, agent_id, agent_weights, performance, volatility, turnover):
        """
        Intenta añadir un agente al archivo si supera al ocupante previo de la celda.
        """
        idx = self._get_grid_index(volatility, turnover)
        
        if idx not in self.archive or performance > self.archive[idx]['performance']:
            self.archive[idx] = {
                'agent_id': agent_id,
                'agent_weights': agent_weights,
                'performance': performance,
                'volatility': volatility,
                'turnover': turnover
            }
            return True # Éxito: Nueva elite encontrada
        return False

    def get_elite_consensus(self):
        """
        Extrae la media de las políticas de la élite actual.
        """
        if not self.archive:
            return None
        
        # En una versión avanzada, aquí se promediarían los pesos neuronales 
        # o se seleccionaría el top N% de celdas.
        performances = [v['performance'] for v in self.archive.values()]
        return np.mean(performances)

if __name__ == "__main__":
    map_e = MAPElites()
    print("Sistema MAP-Elites inicializado para Calidad-Diversidad.")
    map_e.update_archive("Agente_001", None, 150.5, 0.4, 0.6)
    print(f"Agentes en el archivo: {len(map_e.archive)}")
