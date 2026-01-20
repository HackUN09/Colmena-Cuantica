# PROYECTO COLMENA CUÁNTICA: ESPECIFICACIÓN TÉCNICA (V1.0)

> "Arquitectura Zero-Touch para Entornos de Alta Frecuencia."

## 1. Topología del Sistema (Container Stack)

El sistema se despliega sobre una malla de contenedores Docker orquestados para minimizar latencia y maximizar aislamiento.

```mermaid
graph TD
    A[Internet / Binance API] -->|HTTPS / WSS| B(n8n Orchestrator);
    B -->|Trigger 1m| C{Main API (FastAPI)};
    
    subgraph "Colmena Core (GPU)"
    C -->|Fractal Tensor| D[GymEngine];
    D -->|Inference| E[Swarm Agents (100x)];
    E -->|Action| F[Treasury Manager];
    end
    
    F -->|SQL Transaction| G[(PostgreSQL)];
    F -->|Virtual Wallet Update| G;
    
    subgraph "Persistence Layer"
    G -->|Backup| H[Volume: pgdata];
    C -->|Disk I/O| I[Volume: models/];
    end
```

## 2. Flujo de Datos (Data Pipeline)

### 2.1 Ingesta (Fase 0)
*   **Fuente**: CCXT (Binance).
*   **Formato**: OHLCV (Open, High, Low, Close, Volume).
*   **Frecuencia**: 1 minuto.
*   **Buffer**: `main.py` mantiene una `deque` de longitud 1000 en RAM para evitar lecturas de disco constantes durante la inferencia en caliente.

### 2.2 Pre-procesamiento Tensor (Fase 1)
Antes de tocar la red neuronal, los datos crudos $P_t$ sufren transformaciones irreversibles:
1.  **Log-Returns**: $r_t = \ln(P_t) - \ln(P_{t-1})$.
2.  **Fractional Diff**: `frac_diff_ffd(d=0.4)`.
3.  **Normalización Z-Score**: Rolling window de 30 periodos.

### 2.3 Inferencia Asíncrona (Fase 2)
El `main.py` iterar sobre la población.
*   Si `timestamp % agente.dilation == 0`: El agente se activa.
*   Si no: El agente "duerme" (acción anterior se mantiene o decae).
*   Tiempo de vuelta (Round Trip): < 200ms para 100 agentes en GPU.

## 3. Pila Tecnológica (Stack)

*   **Lenguaje**: Python 3.9+ (Tipado estricto).
*   **Tensor Engine**: PyTorch 2.0 (CUDA support).
*   **Dataframes**: Pandas / NumPy (Vectorización masiva).
*   **Neuro-Evolución**: Custom Implementation (No libraries).
*   **API**: FastAPI (Uvicorn worker).
*   **DB**: PostgreSQL 15 (TimescaleDB ready).

## 4. Seguridad y Robustez

*   **Fail-Safe**: Si la inferencia falla, el sistema retorna `HOLD` (Cash 100%) por defecto.
*   **Atomic Saves**: Los pesos `.pth` se guardan primero como `.tmp` y luego se renombran para evitar corrupción por corte de energía.
*   **Cold Start**: Al reiniciar, el sistema carga `cerebro_checkpoint.pth` y consulta la DB para restaurar los saldos virtuales de los 100 agentes.

---
**Infraestructura**: Dockerized / Linux Alpine
**Author**: System Architect
