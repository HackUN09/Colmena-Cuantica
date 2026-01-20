import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Autoencoder Variacional (VAE) optimizado para señales financieras.
    Comprime un vector de 100 indicadores a un espacio latente de 8 dimensiones.
    """
    def __init__(self, input_dim=10, latent_dim=8):
        super(VAE, self).__init__()
        
        # Codificador (Encoder): 100 -> 64 -> 32 -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        
        # Capas latentes
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder: Latente -> 32 -> 64 -> 100
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Asumimos datos normalizados [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Truco de reparametrización para permitir retropropagación a través de variables aleatorias.
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    dummy_input = torch.randn(32, 100).to(device) # Batch de 32, 100 indicadores
    
    recon, mu, logvar = model(dummy_input)
    print(f"Dimensiones de entrada: {dummy_input.shape}")
    print(f"Dimensiones de reconstrucción: {recon.shape}")
    print(f"Dimensiones del espacio latente (mu): {mu.shape}")
