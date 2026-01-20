# COLMENA CUÁNTICA V1.0 // PROTOCOLO FOURIER (Bio-Spectral Resonance)

> **"La matemática define el terreno; la genética define el caminante."**

Este repositorio aloja la versión **V1.0 (Stable)** del Sistema de Trading Algorítmico Jerárquico "Colmena Cuántica". Esta versión implementa la arquitectura **Bio-Espectral**, donde un enjambre de agentes de autogestión (SAC) opera sobre un **Tensor Fractal** del mercado.

## 🌌 Arquitectura Bio-Espectral (Scientific Spec)

> Ver documentación completa en `/docs`:
> - [📘 Especificación Matemática (MATH_SPEC.md)](docs/MATH_SPEC.md)
> - [🧬 Especificación Biológica (BIO_SPEC.md)](docs/BIO_SPEC.md)
> - [🏗️ Especificación Técnica (TECH_SPEC.md)](docs/TECH_SPEC.md)

### 1. Núcleo Matemático (The Atom)
El sistema abandona el concepto de "frecuencia estática". Utiliza **Diferenciación Fraccionaria** ($\nabla^{0.4}$) e **Hipercubos Tensoriales** ($\mathbb{R}^{27}$) para representar el mercado como un fluido dinámico.

### 2. Ingeniería Fractal (The 3-Layer Tensor)
Cada agente percibe el mercado en tres dimensiones temporales simultáneas (ver `MATH_SPEC` Cap. 1):
*   **Micro ($\mathbf{z}_{micro}$)**: La realidad inmediata (1m).
*   **Meso ($\mathbf{z}_{meso}$)**: La visión subjetiva del agente ($T_i \in [10, 60]$).
*   **Macro ($\mathbf{z}_{macro}$)**: La marea global (4h).

### 3. Inteligencia de Enjambre (Swarm Consensus)
Una población de 100 agentes gobernada por un **Algoritmo Genético de Estado Estacionario** (ver `BIO_SPEC` Cap. 2). La colmena elimina a los agentes disonantes y clona a los resonantes.

## 📂 Estructura del Proyecto

```
Gym_trading/
├── data/
│   └── historical/       # "El Átomo": Velas de 1 minuto (Top 10)
├── docker/               # Infraestructura de Contenedores
├── models/               # "La Memoria": Checkpoints neuronales (.pth)
├── src/
│   ├── execution/        # Risk Manager & Treasury (Ledger real)
│   ├── math_kernel/      # Spectral Analysis, Stationarity (FFD), Universe
│   ├── swarm_brain/      # SAC Agents, Genetics, Swarm Controller
│   ├── training/         # GymEngine (Entrenamiento Fractal)
│   ├── vae_layer/        # VAE & NLP Engine (Compresión Latente)
│   └── utils/            # Herramientas auxiliares
├── main.py               # "El Corazón": API Fractal de Producción (FastAPI)
├── monitor_tesoro.py     # Interfaz de Vigilancia (Terminal)
├── sistema_maestro.sh    # "God Mode": Orquestador de Comandos
├── harvest_top10.py      # Herramienta de Ingesta de Datos
└── requirements.txt      # Dependencias (Torch, NumPy, SciPy)
```

## 🚀 Despliegue (Production Ready)

### 1. Iniciar Infraestructura
```bash
./sistema_maestro.sh
# Opción 1: Iniciar Contenedores (Docker)
```

### 2. Ingesta de Datos (Big Bang)
Si el sistema está vacío, descarga el universo base:
```bash
# Dentro del contenedor o via sistema_maestro
python harvest_top10.py
```

### 3. Entrenamiento (Génesis)
Entrena a la Generación 0 para que aprenda a ver en 3D:
```python
# python main.py (Modo Autoservicio)
# O ejecutar GymEngine manualmente para entrenamiento intensivo
from src.training.gym_engine import GymEngine
gym = GymEngine()
gym.train_portfolio(iterations=10)
```

### 4. Conexión n8n (El Flujo)
Importa el flujo `workflow_fourier_v10.json` en tu instancia de n8n.
- **Cron**: 1 minuto.
- **Trigger**: Descarga Top 10 de Binance.
- **Proceso**: Envía datos a `main.py` -> Inferencia Fractal -> SQL Update.

## ⚠️ Hard Fork Warning
Esta versión **V1.0** es incompatible con cerebros anteriores a la Era Fourier.
- **Requiere**: Reset de Tesorería.
- **Requiere**: Nuevos pesos (`state_dim=27`).

---
**Status**: `OPERATIONAL`
**Version**: `1.0 (Fourier Protocol)`
**Author**: `Deepmind & User`
```