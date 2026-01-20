TICKER_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "TRX/USDT", "LINK/USDT"
]

def get_ticker_index(ticker):
    """Retorna el índice estable para un ticker, o -1 si no está en el universo."""
    # Normalizar /
    t = ticker.replace('_', '/')
    if '/' not in t and 'USDT' in t:
        t = f"{t.replace('USDT', '')}/USDT"
    
    try:
        return TICKER_UNIVERSE.index(t)
    except ValueError:
        return -1
