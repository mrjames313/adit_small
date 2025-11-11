
import argparse, os, time, torch, torch.optim as optim
from adit_small.utils import seed_all
from adit_small.data import get_qm9_dataloaders
from adit_small.vae import VAESmall, VAEConfig
from adit_small.dit import DiTConfig, LatentDiT, flow_matching_loss

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--vae_ckpt', type=str, default='./ckpts/vae.pt')
    p.add_argument('--save', type=str, default='./ckpts/dit.pt')
    p.add_argument('--grad_accum', type=int, default=1)
    args = p.parse_args()

    seed_all(42)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dl_train, _ = get_qm9_dataloaders(args.data_root, batch_size=args.batch_size)
    # Load VAE
    vae_cfg = VAEConfig()
    vae = VAESmall(vae_cfg).to(device)
    if os.path.exists(args.vae_ckpt):
        state = torch.load(args.vae_ckpt, map_location=device)
        vae.load_state_dict(state["model"], strict=False)
        print(f"Loaded VAE from {args.vae_ckpt}")
    vae.eval(); 
    for p_ in vae.parameters(): p_.requires_grad=False

    dit_cfg = DiTConfig(d_latent=vae_cfg.d_latent, d_model=256, n_heads=4, n_layers=6)
    dit = LatentDiT(dit_cfg).to(device)
    opt = optim.AdamW(dit.parameters(), lr=args.lr)

    it=0; loss_ema=None
    while it < args.steps:
        for batch in dl_train:
            for k in batch: batch[k]=batch[k].to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(batch["types"], batch["coords"], batch["mask"])
                z1 = vae.reparam(mu, logvar)
                z0 = torch.randn_like(z1)
            t = torch.rand(batch["types"].size(0), device=device)  # (B,)
            loss = flow_matching_loss(dit, z0, z1, t, batch["mask"]) / args.grad_accum
            loss.backward()
            if (it+1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
            it+=1
            loss_ema = loss.item() if loss_ema is None else 0.99*loss_ema+0.01*loss.item()
            if it % 50 == 0:
                print(f"step {it}/{args.steps} loss={loss_ema:.4f}")
            if it>=args.steps: break
    torch.save({"cfg":dit_cfg.__dict__, "model":dit.state_dict()}, args.save)
    print(f"Saved {args.save}")

if __name__ == '__main__':
    main()
