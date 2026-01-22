
import psycopg2
from datetime import datetime
import sys

# Configuración Maestra (Espejo de main.py)
DB_CONFIG = {
    "host": "localhost",
    "database": "n8n",
    "user": "n8n_user",
    "password": "n8n_pass",
    "port": "5433" 
}

def genesis_reset():
    """
    PURGA Y REINICIO DE MATRIZ ECONÓMICA.
    """
    try:
        print("\n[DB MAINTENACE] Iniciando protocolo de purga...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # 1. Limpiar Tablas
        print("[DB MAINTENACE] Limpiando wallets y transacciones...")
        cur.execute("TRUNCATE transactions, wallets RESTART IDENTITY CASCADE;")
        
        # 2. Inyectar Capital Inicial ($1000 - $10 por agente)
        print("[DB MAINTENACE] Inyectando liquidez inicial ($1000 USDT)...")
        for i in range(100):
            ag_id = f"agente_{i}"
            cur.execute("INSERT INTO wallets (agente_id, balance) VALUES (%s, %s);", (ag_id, 10.0))
            
        # 3. Registrar Agente de Sistema (Para integridad referencial)
        cur.execute("INSERT INTO wallets (agente_id, balance) VALUES (%s, %s);", ("SISTEMA_GENESIS", 0.0))
        
        # 4. Registrar Transacción Inicial (Usando 'saldo_final' corregido)
        cur.execute(
            "INSERT INTO transactions (timestamp, agente_id, pnl_bruto, tax_cosecha, saldo_final) VALUES (%s, %s, %s, %s, %s);",
            (datetime.now(), "SISTEMA_GENESIS", 0.0, 0.0, 1000.0)
        )
        
        conn.commit()
        cur.close()
        conn.close()
        print("[DB MAINTENACE] ÉXITO: Sistema reseteado a $1000.00 USDT.")
        
    except Exception as e:
        print(f"\n[DB ERROR CRÍTICO] No se pudo completar el reset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    genesis_reset()
