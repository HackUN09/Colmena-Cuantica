import os
import psycopg2
import time
from datetime import datetime
from colorama import init, Fore, Style

# Inicializar colores para terminal
init()

def monitor_tesoro():
    """
    Auditoría de Espectro Completo v7.3.
    Muestra los 100 agentes en una rejilla compacta.
    """
    try:
        conn = psycopg2.connect(
            host="localhost", 
            database="n8n",
            user="n8n_user",
            password="n8n_pass",
            port="5433" 
        )
    except Exception as e:
        print(f"{Fore.RED}[ERROR] No se pudo conectar a la DB: {e}{Style.RESET_ALL}")
        return

    try:
        while True:
            with conn.cursor() as cur:
                # 1. Capital Total
                cur.execute("SELECT SUM(balance) FROM wallets;")
                total_capital = cur.fetchone()[0] or 0.0
                
                # 2. Reserva
                cur.execute("SELECT SUM(tax_cosecha) FROM transactions;")
                reserva = cur.fetchone()[0] or 0.0
                
                # 3. Operaciones
                cur.execute("SELECT COUNT(*) FROM transactions;")
                n_ops = cur.fetchone()[0]
                
                # 4. Obtener todos los balances (ordenados)
                # 4. Obtener todos los balances (ordenados)
                cur.execute("SELECT agente_id, balance FROM wallets WHERE agente_id LIKE 'agente_%%' ORDER BY CAST(SUBSTRING(agente_id FROM 8) AS INTEGER);")
                all_wallets = cur.fetchall()
                
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════╗")
                print(f"║ {Fore.WHITE}{Style.BRIGHT}AUDITORÍA RADICAL: ESPECTRO DE ENJAMBRE (100 AGENTES)                     {Fore.CYAN}║")
                print(f"╠══════════════════════════════════════════════════════════════════════════╣")
                print(f"║ {Fore.GREEN}STATUS: Sincronizado{Fore.CYAN}   ║ {Fore.WHITE}CAPITAL TOTAL: ${total_capital:>10.2f} USDT{Fore.CYAN}          ║")
                print(f"║ {Fore.YELLOW}RESERVA: ${reserva:>9.6f}{Fore.CYAN} ║ {Fore.MAGENTA}OPERACIONES: {n_ops:>10}{Fore.CYAN}                 ║")
                print(f"╚══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
                
                print(f"\n{Fore.WHITE}DISTRIBUCIÓN DE BILLETERAS (USDT):{Style.RESET_ALL}")
                
                # Dibujar rejilla 10x10 con ALTA RESOLUCIÓN
                for i in range(10):
                    row_str = ""
                    for j in range(10):
                        idx = i * 10 + j
                        if idx < len(all_wallets):
                            bal = float(all_wallets[idx][1])
                            # Color basado en Profit/Loss MICRO
                            color = Fore.WHITE
                            if bal > 10.00000: color = Fore.GREEN
                            elif bal < 9.99999: color = Fore.RED
                            
                            row_str += f"{color}{bal:>10.7f} " # 7 decimales (Alta Sensibilidad)
                        else:
                            row_str += f"{Fore.LIGHTBLACK_EX} ??.????? "
                    print(row_str)
                
                print(f"\n{Fore.LIGHTBLACK_EX}--------------------------------------------------------------------------")
                print(f"Último refresco: {datetime.now().strftime('%H:%M:%S')} | Refresco automático: 60s{Style.RESET_ALL}")
                
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[SISTEMA] Cerrando monitor de auditoría...{Style.RESET_ALL}")
        conn.close()

if __name__ == "__main__":
    monitor_tesoro()
