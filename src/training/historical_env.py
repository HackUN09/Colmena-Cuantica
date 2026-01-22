"""
COLMENA-CUÁNTICA v11.0 - Historical Market Environment
=======================================================
Entorno de simulación offline que replica el comportamiento de mercados reales
usando datos históricos. Permite entrenar agentes 1000× más rápido que en tiempo real.

Fundamento Matemático:
- Proceso de Markov de decisión: (S, A, P, R, γ)
- S: Estados del mercado (OHLCV histórico)
- A: Acciones (pesos de portfolio)  
-P: Dinámica determinista (datos históricos)
- R: Recompensas (P&L con comisiones)
- γ: Factor de descuento (0.99)

Uso:
    env = HistoricalMarketEnv()
    state = env.reset()
    next_state, reward, done = env.step(actions)
"""

import pandas as pd
import numpy as np
import os
import random
from typing import Dict, Tuple, List
from src.math_kernel.ticker_universe import TICKER_UNIVERSE

class HistoricalMarketEnv:
    """
    Entorno de simulación basado en datos históricos.
    Replica exactamente el flujo de main.py pero sin esperar ticks en tiempo real.
    """
    
    def __init__(self, data_path: str = 'data/historical', commission_rate: float = 0.0015):
        """
        Args:
            data_path: Directorio con CSVs históricos
            commission_rate: Tasa de comisión por operación (0.15% por defecto)
        """
        self.data_path = data_path
        self.commission_rate = commission_rate
        self.data = {}  # {symbol: DataFrame}
        self.max_ticks = 0
        self.current_tick = 0
        
        self._load_data()
        
    def _load_data(self):
        """Carga todos los CSVs históricos en memoria"""
        print(f"[ENV] Cargando datos históricos desde {self.data_path}...")
        
        for symbol in TICKER_UNIVERSE[:10]:  # Top 10
            filename = symbol.replace("/", "_") + ".csv"
            filepath = os.path.join(self.data_path, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No existe {filepath}. Ejecutar download_historical.py primero.")
                
            df = pd.read_csv(filepath)
            self.data[symbol] = df
            
            print(f"  ✓ {symbol}: {len(df):,} ticks")
            
        # Encontrar longitud mínima (todos deben tener los mismos timestamps)
        lengths = [len(df) for df in self.data.values()]
        self.max_ticks = min(lengths) - 100  # Reserve 100 para lookahead
        
        print(f"[ENV] ✅ {len(self.data)} activos cargados | Max ticks: {self.max_ticks:,}")
        
    def reset(self, random_start: bool = True) -> Dict:
        """
        Reinicia el entorno a un punto en el tiempo.
        
        Args:
            random_start: Si True, empieza en tick aleatorio (data augmentation)
                         Si False, empieza en tick 0
                         
        Returns:
            Dict con datos de mercado del tick inicial
        """
        if random_start:
            # Empezar en punto aleatorio para diversidad de entrenamiento
            self.current_tick = random.randint(0, max(0, self.max_ticks - 1000))
        else:
            self.current_tick = 0
            
        return self._get_market_data()
        
    def _get_market_data(self) -> Dict:
        """
        Retorna datos de mercado en el formato esperado por main.py
        
        Returns:
            Dict: {ticker: [[timestamp, open, high, low, close, volume]]}
        """
        market_data = {}
        
        for symbol in TICKER_UNIVERSE[:10]:
            row = self.data[symbol].iloc[self.current_tick]
            
            # Formato OHLCV como lista (compatible con ccxt)
            ohlcv = [[
                int(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ]]
            
            market_data[symbol] = ohlcv
            
        return market_data
        
    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool]:
        """
        Ejecuta un paso en el entorno.
        
        Args:
            actions: Array de pesos [w_BTC, w_ETH, ..., w_CASH] (11 dimensiones)
                    Debe sumar 1.0 (100% del capital)
                    
        Returns:
            next_state: Datos de mercado del siguiente tick
            reward: P&L neto (después de comisiones)
            done: Si llegó al final de los datos
        """
        # Validar acciones
        assert len(actions) == 11, f"Expected 11 actions, got {len(actions)}"
        assert abs(sum(actions) - 1.0) < 0.01, f"Actions must sum to 1.0, got {sum(actions)}"
        
        # Obtener precios actuales y siguientes
        current_prices = self._get_prices(self.current_tick)
        next_prices = self._get_prices(self.current_tick + 1)
        
        # Calcular P&L
        pnl = 0.0
        volume_traded = 0.0
        
        for i, symbol in enumerate(TICKER_UNIVERSE[:10]):
            weight = actions[i]
            
            # Retorno del activo
            price_return = (next_prices[symbol] / current_prices[symbol]) - 1.0
            
            # Contribución al P&L (asumiendo balance = 1.0 por simplicidad)
            pnl += weight * price_return
            
            # Volumen operado (para comisiones)
            volume_traded += abs(weight)
            
        # Aplicar comisión (fricción realista)
        commission = volume_traded * self.commission_rate
        pnl_net = pnl - commission
        
        # Avanzar tick
        self.current_tick += 1
        done = self.current_tick >= self.max_ticks
        
        # Obtener siguiente estado
        next_state = self._get_market_data() if not done else {}
        
        return next_state, pnl_net, done
        
    def _get_prices(self, tick: int) -> Dict[str, float]:
        """
        Obtiene precio de cierre de todos los activos en un tick específico.
        
        Returns:
            Dict: {symbol: close_price}
        """
        prices = {}
        for symbol in TICKER_UNIVERSE[:10]:
            row = self.data[symbol].iloc[tick]
            prices[symbol] = float(row['close'])
        return prices
        
    def get_historical_window(self, lookback: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Retorna ventana histórica para análisis (útil para features temporales).
        
        Args:
            lookback: Cuántos ticks hacia atrás obtener
            
        Returns:
            Dict: {symbol: DataFrame con últimos 'lookback' ticks}
        """
        start_tick = max(0, self.current_tick - lookback)
        end_tick = self.current_tick
        
        windows = {}
        for symbol in TICKER_UNIVERSE[:10]:
            windows[symbol] = self.data[symbol].iloc[start_tick:end_tick]
            
        return windows
        
    def get_info(self) -> Dict:
        """Retorna información sobre el estado actual del entorno"""
        return {
            'current_tick': self.current_tick,
            'max_ticks': self.max_ticks,
            'progress': self.current_tick / self.max_ticks,
            'symbols': list(self.data.keys()),
            'commission_rate': self.commission_rate
        }


if __name__ == "__main__":
    # Test básico
    print("=" * 70)
    print("Test: HistoricalMarketEnv")
    print("=" * 70)
    
    env = HistoricalMarketEnv()
    
    # Test 1: Reset
    print("\n[TEST 1] Reset del entorno...")
    state = env.reset(random_start=False)
    print(f"  ✓ Estado inicial obtenido para {len(state)} activos")
    
    # Test 2: Step con acciones aleatorias
    print("\n[TEST 2] Ejecutar 100 steps...")
    total_pnl = 0.0
    
    for i in range(100):
        # Acciones aleatorias (portfolio aleatorio)
        actions = np.random.dirichlet(np.ones(11), size=1)[0]  # Suma a 1.0
        
        next_state, reward, done = env.step(actions)
        total_pnl += reward
        
        if i % 20 == 0:
            print(f"  Tick {i}: Reward = {reward:.6f}, PnL acumulado = {total_pnl:.6f}")
            
        if done:
            print(f"  ⚠️  Llegó al final de los datos en tick {i}")
            break
    
    # Test 3: Info
    print("\n[TEST 3] Información del entorno:")
    info = env.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Todos los tests pasaron exitosamente")
