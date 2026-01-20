import numpy as np
import torch # Usamos PyTorch para aceleración por GPU

def gpu_monte_carlo_sim(current_price, mu, sigma, n_sims=10000, n_days=30):
    """
    Ejecuta simulaciones de Monte Carlo aceleradas por GPU para proyectar trayectorias de precios.
    Retorna el valor esperado y métricas de riesgo (VaR/Upside).
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convertir a tensores de GPU
        dt = 1/252 # Escala diaria (suponiendo trading anual)
        mu_tensor = torch.tensor(mu, device=device)
        sigma_tensor = torch.tensor(sigma, device=device)
        
        # Generar retornos aleatorios normales en bloque (Memoria eficiente)
        # dS = S * (mu*dt + sigma*epsilon*sqrt(dt))
        epsilon = torch.randn((n_sims, n_days), device=device)
        
        # Simulación masiva
        returns = (mu_tensor - 0.5 * sigma_tensor**2) * dt + sigma_tensor * epsilon * np.sqrt(dt)
        cumulative_returns = torch.exp(torch.cumsum(returns, dim=1))
        
        price_trajectories = current_price * cumulative_returns
        
        # Retornar percentiles finales (Riesgo/Oportunidad)
        final_prices = price_trajectories[:, -1]
        results = {
            'expected_value': torch.mean(final_prices).item(),
            'var_95': torch.quantile(final_prices, 0.05).item(),
            'upside_95': torch.quantile(final_prices, 0.95).item()
        }
        return results
    except Exception as e:
        print(f"Error en MC GPU: {e}")
        return None

if __name__ == "__main__":
    print(f"Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    res = gpu_monte_carlo_sim(50000, 0.15, 0.30, n_sims=100000)
    if res:
        print("\nResultados Simulación Monte Carlo (100k sims):")
        print(f"Valor Esperado: {res['expected_value']:.2f}")
        print(f"VaR (95%): {res['var_95']:.2f}")
        print(f"Upside (95%): {res['upside_95']:.2f}")
