
# All Atom diffusion transformer - small implementation

Generative model of atomic / molecular structures.

Simple implementation of diffusion-based unified (periodic and non-periodic molecular structures) model as described in [this paper](https://arxiv.org/pdf/2503.03965).  Includes modeling of atom types, 3d coordinates for each atom, and specification of periodic structures when necessary (out of scope for the dataset used here).

The major components include a Variational Autoencoder (VAE) to map atomic structure to/from a shared (across multiple molecules / crystals) latent space, and a small Diffusion Transformer (DiT) acting on the latent space for the generative modeling.

We use a small, non-periodic molecules only, dataset called QM9 and small VAE / DiT models in order to fit on commodity HW (and on colab-free).

A small, Colab-friendly starter for an ADiT-style pipeline on **QM9** only:
- **Stage A:** VAE learns a per-atom latent (small bottleneck) to reconstruct atom types + 3D coordinates.
- **Stage B:** A small latent **Diffusion Transformer** (DiT) is trained with **flow matching** in the latent space.

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
