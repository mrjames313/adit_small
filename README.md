
# adit-small (Phase 1)

A small, Colab-friendly starter for an ADiT-style pipeline on **QM9** only:
- **Stage A:** VAE learns a per-atom latent (small bottleneck) to reconstruct atom types + 3D coordinates.
- **Stage B:** A tiny latent **Diffusion Transformer** (DiT) is trained with **flow matching** in the latent space.
- **Goal:** End-to-end demo on Colab Free. Expect toy-scale results only.

## Quickstart (Colab)
1. Open the notebook at `notebooks/01_qm9_quickstart.ipynb` in Google Colab.
2. Run the **Setup** cell to install deps and mount Google Drive.
3. Train VAE (~15–45 min on T4), then train DiT (short run). Checkpoints are saved to Drive.

## Local (or cloud) usage
```bash
pip install -e .
python scripts/train_vae.py --epochs 3
python scripts/train_dit.py --steps 1000
```
Use `--help` on each script for options.

## Notes
- Uses `torch_geometric` QM9 if available; otherwise falls back to a small synthetic toy set.
- The architecture is intentionally **simple** and not equivariant. We rely on augmentations.
- This is meant as a scaffold you can scale up on Colab Pro or multi-GPU later.
