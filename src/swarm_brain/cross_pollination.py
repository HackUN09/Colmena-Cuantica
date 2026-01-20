import torch
import copy

def cross_pollinate(agent_a, agent_b, tau=0.01):
    """
    Realiza una actualización suave (Soft Update) entre dos agentes vecinos.
    Esto permite que las estrategias exitosas se propaguen porel enjambre.
    theta_a = (1-tau)*theta_a + tau*theta_b
    """
    with torch.no_grad():
        for param_a, param_b in zip(agent_a.policy.parameters(), agent_b.policy.parameters()):
            param_a.data.copy_((1.0 - tau) * param_a.data + tau * param_b.data)

def calculate_policy_distance(agent_a, agent_b, state_sample):
    """
    Calcula la distancia coseno entre las acciones de dos agentes ante el mismo estado.
    Sirve para definir quiénes son 'vecinos cercanos'.
    """
    with torch.no_grad():
        act_a = agent_a.policy.sample(state_sample)
        act_b = agent_b.policy.sample(state_sample)
        
        cos = torch.nn.CosineSimilarity(dim=-1)
        sim = cos(act_a, act_b)
        return 1.0 - sim.mean().item()

if __name__ == "__main__":
    print("Capacidad de Polinización Cruzada (Swarm Graph) lista.")
