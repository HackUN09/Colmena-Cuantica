import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    Red Actor: Emite la media y desviación estándar para la distribución de acciones de cartera.
    Entrada: Estado Completo v12.0 = 51 dims:
        - VAE Latents (27): micro(8) + meso(8) + macro(8) + sentiment(3)
        - Math Features (16): spectral(2) + physics(4) + prob(4) + stats(2) + linalg(2) + signals(2)
        - Self-Awareness (5): balance, pnl, sharpe, streak, win_rate
        - Swarm Collective (3): bull_ratio, avg_pnl, rank
    Salida: 10 Tickers + 1 Cash = 11 dims.
    """
    def __init__(self, state_dim=51, action_dim=11): 
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # Reparameterization trick
        # Temperature Scaling: T baja (<1.0) afila la distribución (más confianza)
        # T alta (>1.0) la suaviza (más entropía/duda).
        # Usamos T=0.05 para forzar decisiones fuertes en un espacio de 10 activos.
        temperature = 0.05 
        action = torch.softmax(x_t / temperature, dim=-1) # Asegurar que sum(pesos) = 1.0
        return action

class SoftActorCritic:
    """
    Agente individual usando SAC con estado completo de 51 dimensiones.
    """
    def __init__(self, state_dim=51, action_dim=11, lr=3e-4, time_dilation=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.time_dilation = time_dilation # Genética Temporal (Bio-Clock)

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            action = self.policy.sample(state)
        return action.cpu().numpy()[0]

    def update(self, state, action, reward):
        """
        Actualización de la política (Simplificada para el flujo operativo).
        Optimiza la red basándose en el refuerzo recibido del mercado.
        """
        # Asegurar tensores en el dispositivo correcto
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).view(1, -1)
        reward_tensor = torch.as_tensor([reward], dtype=torch.float32, device=self.device)

        # Calculamos la acción que el motor habría tomado
        mu, log_std = self.policy(state)
        predicted_action = self.policy.sample(state)

        # Pérdida: Diferencia entre lo que hizo y el beneficio (Heurística de Gradiente)
        # Premiamos acciones que llevaron a PnL positivo
        loss = F.mse_loss(predicted_action, action_tensor) * (-reward_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == "__main__":
    agent = SoftActorCritic()
    dummy_state = torch.randn(51)  # Estado completo v12.0
    action = agent.select_action(dummy_state)
    print(f"Estado dim: 51, Acción dim: 11")
    print(f"Suma de pesos de la acción: {action.sum():.2f}")
    print(f"Muestra de pesos (top 5): {action[:5]}")
