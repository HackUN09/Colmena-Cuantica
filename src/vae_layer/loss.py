import torch
import torch.nn.functional as F

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Calcula la pérdida total del VAE.
    Loss = Error de Reconstrucción + beta * KL Divergence
    """
    # 1. Error de Reconstrucción (MSE para datos numéricos)
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL Divergence: Fuerza el espacio latente a seguir una distribución N(0, 1)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD

if __name__ == "__main__":
    # Test simple
    recon = torch.randn(10, 100)
    target = torch.randn(10, 100)
    mu = torch.zeros(10, 8)
    logvar = torch.zeros(10, 8)
    
    loss = vae_loss_function(recon, target, mu, logvar)
    print(f"Pérdida calculada: {loss.item():.4f}")
