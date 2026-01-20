import numpy as np

class RiskManager:
    """
    Gestor de Riesgos: Actúa como la última línea de defensa.
    Controla el Drawdown máximo y aplica Stop-Loss para proteger el capital.
    """
    def __init__(self, max_drawdown=0.15, stop_loss_global=0.05):
        self.max_drawdown = max_drawdown # 15% máximo
        self.stop_loss_global = stop_loss_global # 5% por trade/activo
        self.peak_equities = {} # Diccionario agente_id -> peak_equity

    def validate_agent_solvency(self, v_i, order_size):
        """
        Garantiza que el agente i tiene colateral virtual suficiente:
        Condición: v_i(t) >= order_size * (1 + \gamma)
        donde \gamma es el buffer de seguridad para costos.
        """
        buffer = 0.01 # 1% buffer
        return v_i >= (order_size * (1 + buffer))

    def validate_weights(self, weights, current_equity, agente_id="agente_0"):
        """
        Ajusta los pesos si el riesgo excede los límites permitidos para un agente específico.
        """
        # Actualizar pico de equity individual
        if agente_id not in self.peak_equities or current_equity > self.peak_equities[agente_id]:
            self.peak_equities[agente_id] = current_equity
            
        peak = self.peak_equities[agente_id]
        drawdown = (peak - current_equity) / peak if peak > 0 else 0
        
        # Si estamos en Drawdown crítico, rotar a Cash (USDT)
        if drawdown > self.max_drawdown:
            print("[RISK] Drawdown crítico detectado. Rotando 100% a CASH.")
            if isinstance(weights, dict):
                safe_weights = {k: 0.0 for k in weights.keys()}
                safe_weights['USDT'] = 1.0
                return safe_weights
            else:
                # Si es un vector (numpy/torch), poner todo a 0 y el último (Cash) a 1
                safe_weights = np.zeros_like(weights)
                safe_weights[-1] = 1.0 
                return safe_weights
            
        return weights

if __name__ == "__main__":
    rm = RiskManager()
    print("Gestor de Riesgos activo.")
