import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque

class TreasuryManager:
    """
    SINDICATO DE COSECHA: MOTOR DE CONTABILIDAD VIRTUAL Y TESORERÍA CENTRALIZADA.
    
    Formalismo Matemático:
    Sea K_t el capital total en el exchange en el tiempo t.
    Sea N el número de agentes activos.
    Sea v_i(t) el valor de la cartera virtual del agente i en el tiempo t.
    
    1. Condición de Estanqueidad:
       K_t = \sum_{i=1}^{N} v_i(t) + H_t
       donde H_t es el "Fondo de Reserva" (Harvested Fund).
       
    2. Dinámica de la Cartera Virtual:
       dv_i = v_i(t) * [ r_i(dt) - \gamma_i(dt) ] - dH_i
       donde:
       r_i: Retorno generado por la estrategia del agente i.
       \gamma_i: Costos de transacción y slippage.
       dH_i: Fracción de beneficio cosechado hacia el fondo maestro.
    """

    def __init__(self, master_reserve_addr: str, harvest_rate: float = 0.20, commission_rate: float = 0.0015):
        """
        Args:
            master_reserve_addr (str): Dirección o identificador de la moneda de reserva (ej. USDC).
            harvest_rate (float): Tasa de cosecha \eta \in [0, 1]. Por defecto 20%.
            commission_rate (float): Tasa de comisión por operación (ej. 0.0015 = 0.15%).
        """
        self.master_reserve_addr = master_reserve_addr
        self.eta = harvest_rate
        self.comm_rate = commission_rate
        self.agentes_ledger = {} # Dict[agente_id, float]
        self.reserve_fund = 0.0
        self.transaction_log = []
        
        # NUEVOS: Self-awareness tracking
        self.agent_history = { }  # Dict[agente_id, deque[(timestamp, balance, pnl)]]
        self.loss_streaks = {}  # Dict[agente_id, int]
        self.initial_capital = {}  # Dict[agente_id, float]

    def capitalizar_colmena(self, total_capital: float, n_agentes: int):
        """
        Distribuye el capital inicial de forma equitativa (Uniform Distribution).
        v_i(0) = K_0 / N
        """
        if n_agentes <= 0: return
        
        individual_share = total_capital / n_agentes
        for i in range(n_agentes):
            agent_id = f"agente_{i}"
            self.agentes_ledger[agent_id] = individual_share
            self.agent_history[agent_id] = deque(maxlen=100)  # Últimos 100 trades
            self.loss_streaks[agent_id] = 0
            self.initial_capital[agent_id] = individual_share
            
        print(f"[TREASURY] Colmena proyectada: {n_agentes} carteras de {individual_share:.2f} USDT.")

    def procesar_cierre_orden(self, agente_id: str, pnl_neto: float, volume_usd: float = 0.0):
        """
        Actualiza la cartera virtual y ejecuta la 'Cosecha de Beneficios' y 'Comisión de Binance'.
        
        Dinámica:
        1. Descontar Comisión: pnl_bruto = pnl_neto - (volume_usd * comm_rate)
        2. Cosechar si es positivo.
        3. Actualizar Saldo.
        """
        if agente_id not in self.agentes_ledger:
            self.agentes_ledger[agente_id] = 10.0 # Saldo base estándar

        # 0. Aplicar Comisión de Fricción (Realismo Binance++)
        comision = volume_usd * self.comm_rate
        pnl_bruto = pnl_neto - comision
        # 1. Calcular Cosecha (Solo si hay beneficio positivo después de comisión)
        cosecha = 0.0
        if pnl_bruto > 0:
            cosecha = pnl_bruto * self.eta
            self.reserve_fund += cosecha
            
        # 2. Actualizar saldo virtual (Descontando la cosecha)
        pnl_retenido = pnl_bruto - cosecha
        self.agentes_ledger[agente_id] += pnl_retenido
        
        # 3. NUEVO: Actualizar self-awareness metrics
        timestamp = datetime.now().timestamp()
        new_balance = self.agentes_ledger[agente_id]
        self.agent_history.setdefault(agente_id, deque(maxlen=100)).append(
            (timestamp, new_balance, pnl_retenido)
        )
        
        # Update loss streak
        if pnl_retenido < 0:
            self.loss_streaks[agente_id] = self.loss_streaks.get(agente_id, 0) + 1
        else:
            self.loss_streaks[agente_id] = 0
        
        # 4. Registrar en log de auditoría
        self.transaction_log.append({
            'timestamp': datetime.now().isoformat(),
            'agente_id': agente_id,
            'pnl_bruto': pnl_neto,
            'cosecha': cosecha,
            'saldo_final_virtual': self.agentes_ledger[agente_id]
        })
        
        return {
            "auditoria_tesoro": {
                'pnl_retenido': pnl_retenido,
                'cosecha_enviada_a_reserva': cosecha,
                'nuevo_saldo_agente': self.agentes_ledger[agente_id]
            }
        }

    def obtener_resumen_enjambre(self) -> Dict:
        """
        Retorna las métricas agregadas de la economía del enjambre.
        """
        total_virtual = sum(self.agentes_ledger.values())
        return {
            'total_capital_activo': total_virtual,
            'total_reserva_cosechada': self.reserve_fund,
            'coeficiente_eficiencia': self.reserve_fund / (total_virtual + self.reserve_fund + 1e-9),
            'n_agentes_activos': len(self.agentes_ledger)
        }

    def verificar_solvencia(self, agente_id: str, monto_requerido: float) -> bool:
        """
        Verifica si un agente tiene colateral virtual suficiente para una nueva posición.
        Condición: v_i(t) >= monto_requerido
        """
        return self.agentes_ledger.get(agente_id, 0.0) >= monto_requerido
