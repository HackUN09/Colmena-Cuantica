from src.training.gym_engine import GymEngine
import time

def ejecutar_genesis():
    print("=================================================")
    print("       🚀 PROTOCOLO GÉNESIS: FASE DE ENTRENAMIENTO 🚀")
    print("=================================================")
    print("Iniciando el Motor Bio-Espectral (GymEngine)...")
    
    # Instanciar el gimnasio con 100 agentes
    gym = GymEngine(n_agents=100)
    
    # Cargar datos (Top 10)
    prices, ffd = gym.load_portfolio()
    if prices is None:
        print("[ERROR] No se encontraron datos. Ejecuta 'python harvest_top10.py' primero.")
        return

    print(f"[DATA] Tensor de Entrenamiento: {prices.shape} (Timeframes Fractal Ready)")
    
    # Iniciar ciclo de 10 Generaciones
    start_time = time.time()
    gym.train_portfolio(iterations=10)
    end_time = time.time()
    
    print("\n=================================================")
    print(f"✅ GÉNESIS COMPLETADO en {(end_time - start_time)/60:.2f} minutos.")
    print("El cerebro 'cerebro_colmena_entrenado.pth' ha sido forjado.")
    print("Ahora puedes iniciar el sistema maestro en modo PRODUCCIÓN.")
    print("=================================================")

if __name__ == "__main__":
    ejecutar_genesis()
