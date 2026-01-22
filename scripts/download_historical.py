"""
COLMENA-CUÁNTICA v11.0 - Historical Data Downloader
====================================================
Descarga 6 meses de datos históricos de Top 10 criptomonedas para offline training.

Fundamento Matemático:
- Necesitamos ~260,000 ticks (6 meses × 30 días × 24 horas × 60 minutos)
- Para 100 agentes con 78K parámetros cada uno
- Ratio óptimo: 10× parámetros → 780K muestras (3 meses de operación)

Uso:
    python download_historical.py
    
Output:
    data/historical/BTC_USDT.csv
    data/historical/ETH_USDT.csv
    ...
"""

import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
from src.math_kernel.ticker_universe import TICKER_UNIVERSE

# Configuración
DATA_DIR = "data/historical"
START_DATE = '2023-07-01T00:00:00Z'  # 6 meses atrás
TIMEFRAME = '1m'  # Velas de 1 minuto
BATCH_SIZE = 1000  # Límite de ccxt.fetch_ohlcv

def ensure_data_dir():
    """Crear directorio si no existe"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[SETUP] Directorio: {DATA_DIR}")

def download_historical_data(symbol, exchange):
    """
    Descarga datos históricos para un símbolo específico.
    
    Args:
        symbol (str): Par de trading (ej: 'BTC/USDT')
        exchange (ccxt.Exchange): Instancia de exchange
        
    Returns:
        pd.DataFrame: Datos OHLCV con columnas [timestamp, open, high, low, close, volume]
    """
    print(f"\n[DESCARGANDO] {symbol}")
    
    since = exchange.parse8601(START_DATE)
    all_ohlcv = []
    
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=BATCH_SIZE)
            
            if not ohlcv:
                print(f"  ⚠️  No hay más datos disponibles")
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Actualizar timestamp para siguiente batch
            since = ohlcv[-1][0] + 60000  # +1 minuto en milisegundos
            
            # Progreso
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"  → {len(all_ohlcv):,} ticks | Último: {current_date.strftime('%Y-%m-%d %H:%M')}", end='\r')
            
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            break
    
    print()  # Nueva línea después del progreso
    
    # Convertir a DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convertir timestamp a datetime legible
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def save_to_csv(df, symbol):
    """Guardar DataFrame a CSV"""
    filename = symbol.replace("/", "_") + ".csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    df.to_csv(filepath, index=False)
    
    # Estadísticas
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  ✅ Guardado: {filepath}")
    print(f"     Filas: {len(df):,} | Tamaño: {size_mb:.2f} MB")
    print(f"     Rango: {df['datetime'].min()} → {df['datetime'].max()}")

def main():
    """Función principal"""
    print("=" * 70)
    print("COLMENA-CUÁNTICA v11.0 - Historical Data Downloader")
    print("=" * 70)
    
    ensure_data_dir()
    
    # Inicializar exchange
    print("\n[CONECTANDO] Binance API...")
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Respetar límites de API
        'options': {'defaultType': 'spot'}
    })
    
    # Descarga para cada ticker
    total_ticks = 0
    successful = 0
    
    for symbol in TICKER_UNIVERSE[:10]:  # Top 10 (ya incluyen "/USDT")
        
        try:
            df = download_historical_data(symbol, exchange)
            
            if len(df) > 0:
                save_to_csv(df, symbol)
                total_ticks += len(df)
                successful += 1
            else:
                print(f"  ⚠️  No hay datos para {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error fatal en {symbol}: {e}")
            continue
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Activos descargados: {successful}/{len(TICKER_UNIVERSE[:10])}")
    print(f"Total de ticks: {total_ticks:,}")
    print(f"Promedio por activo: {total_ticks // max(successful, 1):,}")
    
    # Estimación de utilidad
    samples_per_agent = total_ticks * 100  # 100 agentes
    params_per_agent = 78614
    ratio = samples_per_agent / params_per_agent
    
    print(f"\n[ANÁLISIS]")
    print(f"  Muestras totales generables: {samples_per_agent:,}")
    print(f"  Parámetros por agente: {params_per_agent:,}")
    print(f"  Ratio muestras/parámetros: {ratio:.1f}× (óptimo: 10-100×)")
    
    if ratio >= 10:
        print(f"  ✅ Datos SUFICIENTES para entrenamiento robusto")
    elif ratio >= 5:
        print(f"  ⚠️  Datos ACEPTABLES pero considerar más histórico")
    else:
        print(f"  ❌ Datos INSUFICIENTES - añadir más meses")
    
    print("\n✨ Descarga completada. Proceder con train_offline.py")

if __name__ == "__main__":
    main()
