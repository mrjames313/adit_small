
import argparse, os, time, math
import torch, torch.optim as optim
from adit_small.utils import seed_all
from adit_small.data import get_qm9_dataloaders
from adit_small.vae import VAESmall, VAEConfig, vae_loss
from adit_small.metrics import rmsd_batch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='./data')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-4)
    #p.add_argument('--kl_weight', type=float, default=1e-3)
    p.add_argument("--kl_start", type=float, default=0.0, help="initial KL weight β0")
    p.add_argument("--kl_end",   type=float, default=1e-3, help="final KL weight βT")
    p.add_argument("--kl_anneal_steps", type=int, default=2000, help="linear ramp steps from kl_start→kl_end; after this hold at kl_end")
    p.add_argument('--save', type=str, default='./ckpts/vae.pt')
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--cosine", action="store_true")

    # For RMSD alignment using a more precise method
    p.add_argument("--rmsd_align", action="store_true", help="Use Kabsch alignment for RMSD")

    args = p.parse_args()

    seed_all(42)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dl_train, dl_val = get_qm9_dataloaders(args.data_root, batch_size=args.batch_size)
    cfg = VAEConfig()
    model = VAESmall(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(dl_train) // max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    
    def kl_weight_for_step(global_step: int) -> float:
        t = min(1.0, global_step / max(1, args.kl_anneal_steps))
        return args.kl_start + t * (args.kl_end - args.kl_start)

    # compute updates (optimizer steps), not batches
    updates_per_epoch = math.ceil(len(dl_train) / max(1, args.grad_accum))
    total_updates = args.epochs * updates_per_epoch

    # choose warmup in updates; you can keep --warmup_steps or switch to --warmup_frac
    warmup_updates = min(args.warmup_steps, max(1, total_updates // 10))  # e.g., 10% if steps big

    # build schedulers that operate in *updates*
    warm = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1e-3,   # start at 0.1% of args.lr; adjust if you want
        end_factor=1.0,
        total_iters=warmup_updates
    )
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, total_updates - warmup_updates),
        eta_min=args.lr * 0.05   # cosine floor at 5% of base LR; tweak if desired
    )
    if args.cosine:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warm, cos],
            milestones=[warmup_updates]
        )
    else:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warm], milestones=[]
        )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    for ep in range(1, args.epochs+1):
        model.train()
        s=time.time()
        tot=0; n=0
        opt.zero_grad()
        running = 0.0
        for i, batch in enumerate(dl_train):
            batch = {k: v.to(device, non_blocking=True) for k,v in batch.items()}

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(batch)
                beta = kl_weight_for_step(global_step)
                loss, logs = vae_loss(batch, out, kl_weight=beta, type_pad_id=cfg.type_pad_id)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if sched is not None: sched.step()
                #if global_step % 100 == 0:
                #    print(global_step, opt.param_groups[0]["lr"])
            running += loss.item()
            tot += loss.item(); n += 1
        print(f'Epoch {ep} train_loss={tot/max(n,1):.4f} time={time.time()-s:.1f}s | {logs}')

        # quick val
        model.eval()
        with torch.no_grad():
            tot=0; n=0
            rmsd_sum = 0.0; rmsd_n = 0
            for batch in dl_val:
                for k in batch: batch[k]=batch[k].to(device)
                out = model(batch)
                loss,_ = vae_loss(batch, out, kl_weight=args.kl_end, type_pad_id=cfg.type_pad_id)
                tot+=loss.item(); n+=1
                # RMSD (Å). align=True uses Kabsch per molecule.
                r = rmsd_batch(batch["coords"], out["coords_hat"], batch["mask"], align=args.rmsd_align)  # [B]
                rmsd_sum += r.sum().item()
                rmsd_n   += r.numel()
        val_rmsd = rmsd_sum / max(rmsd_n, 1)
        val_loss = tot / max(n,1)
        print(f'Val loss={val_loss:.4f} | Val RMSD={val_rmsd:.4f} Å')
        #print(f'Val loss={tot/max(n,1):.4f}')
        torch.save({"cfg":cfg.__dict__, "model":model.state_dict()}, args.save)
        print(f"Saved {args.save}")

if __name__ == '__main__':
    main()
