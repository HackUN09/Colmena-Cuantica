import ccxt
import pandas as pd
import os
from datetime import datetime
import time

class DataHarvester:
    """
    Componente de descarga masiva para el Protocolo Crónica.
    Recupera datos históricos OHLCV para alimentar el gimnasio de entrenamiento.
    """
    def __init__(self, exchange_id='binance', storage_dir='data/historical'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=1000):
        """
        Descarga datos OHLCV para un símbolo específico.
        """
        print(f"[HARVESTER] Descargando {symbol} ({timeframe})...")
        try:
            # Soportar símbolos con formato Binance (BTCUSDT) o CCXT (BTC/USDT)
            ccxt_symbol = symbol.replace('/', '')
            # Normalizar a formato CCXT si es necesario
            if 'USDT' in ccxt_symbol and '/' not in symbol:
                symbol = f"{ccxt_symbol.replace('USDT', '')}/USDT"

            all_ohlcv = []
            current_since = since
            
            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                # El siguiente 'since' es el timestamp de la última vela + 1ms
                current_since = ohlcv[-1][0] + 1
                
                # Respetar rate limit
                time.sleep(self.exchange.rateLimit / 1000)
                
                # Si hemos llegado al presente o al límite deseado (si lo hubiera)
                if len(ohlcv) < limit:
                    break
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            filename = os.path.join(self.storage_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
            df.to_csv(filename, index=False)
            print(f"[HARVESTER] Guardado: {filename} ({len(df)} velas)")
            return df
            
        except Exception as e:
            print(f"[HARVESTER ERROR] Fallo en {symbol}: {e}")
            return None

    def harvest_swarm_targets(self, symbols, timeframe='1m', days_back=30):
        """
        Descarga datos para una lista de símbolos.
        """
        since = self.exchange.milliseconds() - (days_back * 24 * 60 * 60 * 1000)
        results = {}
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, since=since)
            if df is not None:
                results[symbol] = df
        return results

    def fetch_top_tickers(self, limit=50, quote='USDT'):
        """
        Obtiene los símbolos con mayor volumen en el mercado de spot.
        """
        print(f"[HARVESTER] Buscando los top {limit} tickers en {quote}...")
        markets = self.exchange.fetch_tickers()
        # Filtrar por moneda base (ej: USDT) y ordenar por volumen de 24h
        sorted_tickers = sorted(
            [m for m in markets.values() if m['symbol'].endswith(f"/{quote}")],
            key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0,
            reverse=True
        )
        top_symbols = [t['symbol'] for t in sorted_tickers[:limit]]
        print(f"[HARVESTER] Top tickers encontrados: {top_symbols}")
        return top_symbols

if __name__ == "__main__":
    # Test rápido con BTC
    harvester = DataHarvester()
    harvester.harvest_swarm_targets(['BTC/USDT'], days_back=1)
