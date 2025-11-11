# src/adit_small/metrics.py
import torch

def _masked_center(x, mask):
    # x: [B,N,3], mask: [B,N]
    m = mask.unsqueeze(-1).float()
    count = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # [B,1,1]
    mean = (x * m).sum(dim=1, keepdim=True) / count                   # [B,1,3]
    return (x - mean) * m, m, count

def _kabsch_align(P, Q):
    """
    P, Q: [n, 3] (already centered); returns R s.t. R @ P^T ~ Q^T
    """
    # Compute covariance
    C = P.T @ Q  # [3,3]
    # SVD
    U, S, Vh = torch.linalg.svd(C)
    R = Vh.T @ U.T
    # Proper rotation (det(R)=+1)
    if torch.det(R) < 0:
        Vh[:, -1] *= -1
        R = Vh.T @ U.T
    return R

@torch.no_grad()
def rmsd_batch(coords_true, coords_pred, mask, align=True):
    """
    Returns per-molecule RMSD in Å as [B] tensor.
    If align=True, applies Kabsch rigid alignment per molecule (CPU loop for stability).
    """
    device = coords_true.device
    # Center both (masked)
    P, m, cnt = _masked_center(coords_true, mask)   # [B,N,3], [B,N,1], [B,1,1]
    Q, _, _   = _masked_center(coords_pred, mask)

    B, N, _ = P.shape
    rmsd_vals = torch.zeros(B, device=device, dtype=P.dtype)

    if align:
        # Small, stable per-sample CPU loop (molecules are tiny)
        P_cpu = P.detach().cpu()
        Q_cpu = Q.detach().cpu()
        m_cpu = m.detach().cpu().squeeze(-1).bool()
        out = []
        for b in range(B):
            idx = m_cpu[b]  # [N]
            Pb = P_cpu[b][idx]  # [n,3]
            Qb = Q_cpu[b][idx]
            if Pb.numel() == 0:
                out.append(0.0)
                continue
            R = _kabsch_align(Pb, Qb)  # [3,3]
            Pb_aligned = (Pb @ R.T)
            diff2 = (Pb_aligned - Qb).pow(2).sum(dim=-1)  # [n]
            out.append((diff2.mean()).sqrt().item())
        return torch.tensor(out, device=device, dtype=P.dtype)
    else:
        # No rotation — just centered RMSD
        diff2 = ((P - Q).pow(2) * m).sum(dim=-1)  # [B,N]
        denom = mask.sum(dim=1).clamp(min=1)      # [B]
        return (diff2.sum(dim=-1) / denom).sqrt()
