import torch
import numpy as np
import random
from collections import deque

class SharedReplayBuffer:
    """
    Tarea 21.47: Shared Memory (Experience Replay).
    Buffer circular compartido por todos los agentes del enjambre para
    aprendizaje off-policy (SAC).
    """
    def __init__(self, capacity=100000, state_dim=11, action_dim=101):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def push(self, state, action, reward, next_state, done):
        """
        Tarea 21.48: Insertar experiencia.
        """
        # Asegurar que sean numpy arrays o valores simples para eficiencia en RAM
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()
        if isinstance(action, torch.Tensor): action = action.cpu().numpy()
        
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Tarea 21.49: Muestreo aleatorio (Batch).
        Retorna tensores listos para entrenamiento en GPU.
        """
        batch = random.sample(self.buffer, batch_size)
        
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state)).to(self.device),
            torch.FloatTensor(np.array(action)).to(self.device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_state)).to(self.device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)
