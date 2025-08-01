import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# Conditional VAE Length Generator
# ----------------------------------------
class LengthVAE(nn.Module):
    def __init__(self, cond_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        # encoder: from condition ⇒ distribution over log‑length
        self.enc_fc1 = nn.Linear(cond_dim, hidden_dim)
        self.enc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        # decoder: from latent + condition ⇒ length scalar
        self.dec_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, 1)

    def encode(self, cond):
        h = F.relu(self.enc_fc1(cond))
        mu, logvar = self.enc_mu(h), self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        h = torch.cat([z, cond], dim=-1)
        h = F.relu(self.dec_fc1(h))
        # output unconstrained real; we’ll exponentiate to get positive length
        logL = self.dec_out(h)
        return torch.exp(logL).squeeze(-1)

    def forward(self, cond):
        mu, logvar = self.encode(cond)
        z = self.reparameterize(mu, logvar)
        L_pred = self.decode(z, cond)
        return L_pred, mu, logvar


# ----------------------------------------
# 3. TDFormer + Loss Integration (simplified)
# ----------------------------------------
class DiffTrafficModel(nn.Module):
    def __init__(self, cond_dim, N, tdformer_cfg):
        super().__init__()
        self.N = N
        self.length_vae = LengthVAE(cond_dim)
        self.tdformer   = TDFormer(input_dim=tdformer_cfg['dim'], **tdformer_cfg)
        # you’ll have your own diffusion U-Net or transformer block

    def forward(self, x0, cond):
        # x0: (batch, N, feat_dim)  original resampled to N
        # cond: (batch, cond_dim)
        # 1) reconstruct length
        L_pred, mu, logvar = self.length_vae(cond)
        # 2) embed x0 and add noise (diffusion forward)
        h = self._embed(x0, cond)   # include positional encoding at N points
        # 3) denoise:
        x_t, loss_diff = self.tdformer(h, cond)  # your diffusion step & loss
        # 4) KL for VAE
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        # Total loss
        loss = loss_diff + kld
        return loss, L_pred

if __name__ == "__main__":
    lg_model = LengthVAE(cond_dim=3)
    cond = torch.randn(10, 3)
    L_true = torch.rand(10) 
    L_pred, mu, logvar = lg_model(cond)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    recon_loss = F.mse_loss(L_pred, L_true)
    print("KL Loss:", kld.item())
    print("Recon Loss:", recon_loss.item())