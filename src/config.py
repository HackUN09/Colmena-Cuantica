"""
COLMENA CUÁNTICA V1.0 - Configuración Central
==============================================
Todos los parámetros del sistema en un solo lugar.
"""

# ==================== ECONOMÍA DEL ENJAMBRE ====================

# Tasa de cosecha: % de ganancias que va al fondo de reserva
HARVEST_RATE = 0.20  # 20% (Spartan: 0.25 = 25%)

# Comisión por operación (simula Binance Spot)
COMMISSION_RATE = 0.0015  # 0.15%

# ==================== ARQUITECTURA DE AGENTES ====================

# Número total de agentes en el enjambre
N_AGENTS = 100  # Recomendado: 100-300
                # Más agentes = más diversidad pero más lento
                # 100 agentes: ~7.8M parámetros totales
                # 300 agentes: ~23.4M parámetros

# Dimensionalidad del estado (NO CAMBIAR sin modificar StateBuilder)
STATE_DIM = 51  # 27 VAE + 16 Math + 5 Self + 3 Swarm

# Dimensionalidad de acciones (10 activos + 1 cash)
ACTION_DIM = 11

# ==================== ENTRENAMIENTO OFFLINE ====================

# Número de episodios de entrenamiento
N_EPISODES = 500  # Cada episodio es un "juego completo"
                  # 100 = rápido (~30 min) - prueba
                  # 500 = estándar (~2-3h) - recomendado
                  # 1000 = largo (~5-6h) - para máxima convergencia

# Ticks por episodio (longitud de cada "partida")
TICKS_PER_EPISODE = 1000  # Cuántos ticks de mercado simular por episodio
                          # 1000 ticks ≈ 16 horas de mercado (1 tick = 1 min)

# Intervalo para guardar checkpoints
CHECKPOINT_INTERVAL = 50  # Guardar cada N episodios

# Learning rate para SAC
LEARNING_RATE = 3e-4  # Adam optimizer

# ==================== FEATURES MATEMÁTICAS ====================

# Activar/desactivar módulos del arsenal
ENABLE_SPECTRAL = True      # Fourier, periodicidades
ENABLE_PHYSICS = True        # Hurst, Fractals, Lyapunov
ENABLE_PROBABILITY = True    # HMM, EVT
ENABLE_STATISTICS = True     # GARCH, ARIMA
ENABLE_LINEAR_ALGEBRA = True # RMT, PCA
ENABLE_SIGNALS = True        # Wavelets, Kalman

# Tamaño de ventana para cálculos temporales
FEATURE_WINDOW_SIZE = 100  # Mínimo recomendado: 100

# Cachear computaciones pesadas (HMM, GARCH)
CACHE_HEAVY_COMPUTATIONS = True
CACHE_INTERVAL = 100  # Re-calcular cada N ticks

# ==================== ALGORITMO GENÉTICO ====================

# Porcentaje de agentes que sobreviven cada generación
ELITE_RATIO = 0.10  # Top 10% son "elite"
CULL_RATIO = 0.10   # Bottom 10% son eliminados

# Frecuencia de evolución
EVOLUTION_INTERVAL = 1000  # Cada N ticks en live mode

# ==================== HARDWARE ====================

# Device (detectado automáticamente, pero se puede forzar)
DEVICE = 'cuda'  # 'cuda' o 'cpu'

# Precisión de cálculo
DTYPE = 'float32'  # 'float32' o 'float16' (mixed precision)

# ==================== PATHS ====================

MODELS_DIR = 'models/pretrained'
RESULTS_DIR = 'results/offline_training'
STATS_DIR = RESULTS_DIR  # Alias para compatibilidad
DATA_DIR = 'data/historical'
CHECKPOINTS_DIR = 'models/checkpoints'

# ==================== EXPLICACIÓN DE TÉRMINOS ====================

"""
EPISODIO vs ÉPOCA:
- EPISODIO: Una "partida completa" del agente desde inicio hasta fin.
  En nuestro caso, 1 episodio = simular 1000 ticks de mercado.
  Al final de cada episodio, el agente ve cuánto ganó/perdió.
  
- ÉPOCA (EPOCH): En supervised learning, 1 época = pasar por TODO el dataset una vez.
  En RL no usamos épocas, usamos episodios.

GENERACIÓN (Genetic Algorithm):
- 1 generación = 1 ciclo de selección/cull/breeding
- Ocurre cada EVOLUTION_INTERVAL ticks en live mode
- En offline training, no hay evolución genética (solo RL puro)

AGENTES:
- Más agentes = más diversidad de estrategias
- Pero también más lento (100 agentes: 1x, 300 agentes: ~3x más lento)
- Recomendado: 100 para desarrollo, 200-300 para producción

TIEMPO DE ENTRENAMIENTO:
- RTX 3060 (12GB):
  - 100 agentes × 500 episodios ≈ 2-3 horas
  - 300 agentes × 500 episodios ≈ 6-9 horas
- RTX 4090 (24GB):
  - 100 agentes × 500 episodios ≈ 1 hora
  - 300 agentes × 500 episodios ≈ 3 horas
"""
