
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

@dataclass
class VAEConfig:
    d_model:int=256
    n_heads:int=4
    n_layers:int=4
    d_ff:int=512
    d_latent:int=8   # per-atom latent bottleneck
    max_atoms:int=64
    n_types:int=5
    type_pad_id:int=-1

class TokenEmbed(nn.Module):
    def __init__(self, n_types, d_model):
        super().__init__()
        self.emb = nn.Embedding(n_types, d_model)
    def forward(self, x):
        return self.emb(x.clamp_min(0))  # pad_id is -1; clamp to 0 for embed

class CoordFourier(nn.Module):
    def __init__(self, d_out=64, n_freq=8):
        super().__init__()
        self.n_freq=n_freq
        self.lin = nn.Linear(3*(2*n_freq+1), d_out)
    def forward(self, xyz):
        # xyz: (B,N,3)
        outs=[xyz]
        for k in range(1,self.n_freq+1):
            outs += [torch.sin(k*xyz), torch.cos(k*xyz)]
        feat = torch.cat(outs, dim=-1)
        return self.lin(feat)

def make_transformer_encoder(d_model, n_heads, d_ff, n_layers):
    enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(enc_layer, num_layers=n_layers)

class VAESmall(nn.Module):
    def __init__(self, cfg:VAEConfig):
        super().__init__()
        self.cfg=cfg
        self.tok = TokenEmbed(cfg.n_types, cfg.d_model)
        self.coord = CoordFourier(d_out=cfg.d_model//2)
        self.in_proj = nn.Linear(cfg.d_model + cfg.d_model//2, cfg.d_model)
        self.encoder = make_transformer_encoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers)
        self.mu = nn.Linear(cfg.d_model, cfg.d_latent)
        self.logvar = nn.Linear(cfg.d_model, cfg.d_latent)

        # decoder
        self.dec_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.decoder = make_transformer_encoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers)
        self.type_head = nn.Linear(cfg.d_model, cfg.n_types)
        self.coord_head = nn.Linear(cfg.d_model, 3)

    def encode(self, types, coords, mask):
        x_tok = self.tok(types.clamp_min(0))
        x_pos = self.coord(coords)
        x = torch.cat([x_tok, x_pos], dim=-1)
        x = self.in_proj(x)
        x = self.encoder(x, src_key_padding_mask=~mask)
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, mask):
        h = self.dec_proj(z)
        h = self.decoder(h, src_key_padding_mask=~mask)
        type_logits = self.type_head(h)
        coords = self.coord_head(h)
        return type_logits, coords

    def forward(self, batch):
        types, coords, mask = batch["types"], batch["coords"], batch["mask"]
        mu, logvar = self.encode(types, coords, mask)
        z = self.reparam(mu, logvar)
        type_logits, coords_hat = self.decode(z, mask)
        return {"mu":mu, "logvar":logvar, "z":z, "type_logits":type_logits, "coords_hat":coords_hat}

def vae_loss(batch, out, kl_weight=1e-3, type_pad_id=-1):
    import torch
    import torch.nn.functional as F

    types, coords, mask = batch["types"], batch["coords"], batch["mask"]
    type_logits, coords_hat = out["type_logits"], out["coords_hat"]

    # --- Cross-entropy for atom types over valid atoms ---
    ce = F.cross_entropy(type_logits[mask], types[mask].clamp_min(0))

    # --- Masked zero-centering for coordinates ---
    # shapes: coords                [B, N, 3]
    #         mask                  [B, N]
    # Expand mask to coordinates shape
    m = mask.unsqueeze(-1).float()                  # [B, N, 1]

    # Sum only valid atoms
    coords_sum      = (coords     * m).sum(dim=1, keepdim=True)   # [B, 1, 3]
    coords_hat_sum  = (coords_hat * m).sum(dim=1, keepdim=True)   # [B, 1, 3]

    # Valid counts per example (avoid divide by zero)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # [B, 1, 1]

    # Per-molecule means
    coords_mean     = coords_sum     / counts                          # [B, 1, 3]
    coords_hat_mean = coords_hat_sum / counts                          # [B, 1, 3]

    # Center and keep only valid atoms for MSE
    coords_centered     = (coords     - coords_mean) * m
    coords_hat_centered = (coords_hat - coords_hat_mean) * m

    mse = F.mse_loss(
        coords_hat_centered[mask],  # [sum(N_valid), 3]
        coords_centered[mask]
    )

    # --- KL (per-atom, averaged over valid atoms) ---
    mu, logvar = out["mu"], out["logvar"]           # [B, N, d_latent]
    kl_per_atom = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)  # [B, N]
    kl = (kl_per_atom * mask.float()).sum() / mask.sum().clamp(min=1)

    loss = ce + mse + kl_weight * kl
    return loss, {"ce": ce.item(), "mse": mse.item(), "kl": kl.item(), "kl_w": kl_weight}

