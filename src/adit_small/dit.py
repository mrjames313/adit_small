
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class DiTConfig:
    d_latent:int=8
    d_model:int=256
    n_heads:int=4
    n_layers:int=6

def make_transformer(d_model, n_heads, n_layers, d_ff=512):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=n_layers)

class TimeEmbed(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))
    def forward(self, t):  # t in [0,1], shape (B,1)
        return self.mlp(t)

class LatentDiT(nn.Module):
    def __init__(self, cfg:DiTConfig):
        super().__init__()
        self.cfg=cfg
        self.in_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.time = TimeEmbed(cfg.d_model)
        self.tr = make_transformer(cfg.d_model, cfg.n_heads, cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.d_latent)

    def forward(self, zt, t, mask):
        # zt: (B,N,d_latent); t: (B,1); mask: (B,N)
        h = self.in_proj(zt)
        te = self.time(t)[:,None,:]
        h = h + te
        h = self.tr(h, src_key_padding_mask=~mask)
        v = self.out(h)  # predicted vector field
        return v

def flow_matching_loss(model, z0, z1, t, mask):
    # Linear flow: zt = (1-t) z0 + t z1 ; target u = (z1 - zt)/(1-t)
    zt = (1.0 - t)[:,None,None]*z0 + t[:,None,None]*z1
    with torch.no_grad():
        denom = (1.0 - t).clamp(min=1e-3)
        u = (z1 - zt) / denom[:,None,None]
    v = model(zt, t[:,None], mask)
    loss = F.mse_loss(v[mask], u[mask])
    return loss

@torch.no_grad()
def sample_flow_euler(model, B, N, d_latent, steps=100, device="cuda", mask=None):
    z = torch.randn(B,N,d_latent, device=device)
    t = torch.zeros(B, device=device)
    dt = 1.0/steps
    for s in range(steps):
        tt = t + s*0 + dt*s  # broadcast just to shape properly
        tt = torch.full((B,), s*dt, device=device)
        v = model(z, tt[:,None], mask)
        denom = (1.0 - tt).clamp(min=1e-3)
        z = z + dt * v  # simple Euler step in latent ODE
    return z
