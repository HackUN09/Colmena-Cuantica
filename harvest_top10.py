from src.training.data_harvester import DataHarvester
from src.math_kernel.ticker_universe import TICKER_UNIVERSE

def harvest_fourier_universe():
    print(f"[HARVEST] Iniciando descarga del Universo Fourier (Top 10)...")
    harvester = DataHarvester()
    
    # Descargar 30 días de historia para entrenamiento inicial
    # 30 días * 1440 min/día = ~43k velas 
    # Suficiente para entrenar tendencia de 4h (240 min)
    harvester.harvest_swarm_targets(TICKER_UNIVERSE, timeframe='1m', days_back=30)
    print(f"[HARVEST] Operación completada. Datos listos en data/historical/")

if __name__ == "__main__":
    harvest_fourier_universe()
