import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

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

    def __init__(self, master_reserve_addr: str, harvest_rate: float = 0.20):
        """
        Args:
            master_reserve_addr (str): Dirección o identificador de la moneda de reserva (ej. USDC).
            harvest_rate (float): Tasa de cosecha \eta \in [0, 1]. Por defecto 20%.
        """
        self.master_reserve_addr = master_reserve_addr
        self.eta = harvest_rate
        self.agentes_ledger = {} # Dict[agente_id, float]
        self.reserve_fund = 0.0
        self.transaction_log = []

    def capitalizar_colmena(self, total_capital: float, n_agentes: int):
        """
        Distribuye el capital inicial de forma equitativa (Uniform Distribution).
        v_i(0) = K_0 / N
        """
        if n_agentes <= 0: return
        
        individual_share = total_capital / n_agentes
        for i in range(n_agentes):
            self.agentes_ledger[f"agente_{i}"] = individual_share
            
        print(f"[TREASURY] Colmena proyectada: {n_agentes} carteras de {individual_share:.2f} USDT.")

    def procesar_cierre_orden(self, agente_id: str, pnl_neto: float):
        """
        Actualiza la cartera virtual y ejecuta la 'Cosecha de Beneficios'.
        
        La función de Cosecha H(pnl) se define como:
        H(\Delta P) = \eta \cdot \max(0, \Delta P)
        
        La actualización del saldo virtual es:
        v_i(t+1) = v_i(t) + \Delta P - H(\Delta P)
        """
        if agente_id not in self.agentes_ledger:
            # Si el agente no está inicializado (ej. n8n no mandó el estado), 
            # lo inicializamos con saldo cero para evitar crashes.
            self.agentes_ledger[agente_id] = 100.0 # Saldo base de cortesía

        # 1. Calcular Cosecha (Solo si hay beneficio positivo)
        cosecha = 0.0
        if pnl_neto > 0:
            cosecha = pnl_neto * self.eta
            self.reserve_fund += cosecha
            
        # 2. Actualizar saldo virtual (Descontando la cosecha)
        pnl_retenido = pnl_neto - cosecha
        self.agentes_ledger[agente_id] += pnl_retenido
        
        # 3. Registrar en log de auditoría
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
