import torch
import torch.optim as optim
from .model import VAE
from .loss import vae_loss_function

def train_vae(data_loader, epochs=50, lr=1e-3, device='cuda'):
    """
    Entrena el VAE para aprender a comprimir los indicadores matemáticos de los activos.
    """
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}')
    
    # Guardar pesos
    torch.save(model.state_dict(), 'models/weights/vae_weights.pth')
    return model

if __name__ == "__main__":
    print("Módulo de entrenamiento VAE cargado.")
