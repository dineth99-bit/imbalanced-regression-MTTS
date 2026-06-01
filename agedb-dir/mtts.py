"""
Meta-Learned Target Topology Smoothing (MTTS) for deep imbalanced regression.

Learns a constrained row-stochastic kernel K_theta(b, b') over target bins to smooth
label density, initialized from Gaussian LDS and meta-trained with a bin-balanced objective.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_lds_kernel_window


def build_gaussian_kernel_matrix(num_bins, sigma=2.0, ks=9):
    """Full B x B row-stochastic kernel matching LDS Gaussian smoothing."""
    window = get_lds_kernel_window("gaussian", ks, sigma)
    half = (ks - 1) // 2
    K = np.zeros((num_bins, num_bins), dtype=np.float64)
    for b in range(num_bins):
        row = np.zeros(num_bins)
        for bprime in range(num_bins):
            row[bprime] = window[half + (bprime - b)] if abs(bprime - b) <= half else 0.0
        s = row.sum()
        K[b] = row / s if s > 0 else row
    return K


def compute_raw_bin_counts(labels, num_bins, reweight="sqrt_inv"):
    counts = np.zeros(num_bins, dtype=np.float64)
    for y in labels:
        b = min(num_bins - 1, int(y))
        counts[b] += 1
    if reweight == "sqrt_inv":
        counts = np.sqrt(counts)
    elif reweight == "inverse":
        counts = np.clip(counts, 5, 1000)
    return counts


class MTTSKernel(nn.Module):
    def __init__(
        self,
        num_bins=121,
        embed_dim=32,
        lambda_d=0.1,
        lds_sigma=2.0,
        lds_ks=9,
        beta=1.0,
        epsilon=1e-6,
        meta_tau=5.0,
        residual_scale=0.1,
        unconstrained=False,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.lambda_d = lambda_d
        self.beta = beta
        self.epsilon = epsilon
        self.meta_tau = meta_tau
        self.residual_scale = residual_scale
        self.unconstrained = unconstrained

        K0 = build_gaussian_kernel_matrix(num_bins, sigma=lds_sigma, ks=lds_ks)
        init_logits = np.log(np.clip(K0, 1e-12, None))
        self.register_buffer("init_logits", torch.tensor(init_logits, dtype=torch.float32))
        self.register_buffer("K0", torch.tensor(K0, dtype=torch.float32))
        self.register_buffer(
            "bin_distance",
            torch.tensor(
                np.abs(np.arange(num_bins)[:, None] - np.arange(num_bins)[None, :]),
                dtype=torch.float32,
            ),
        )

        self.embeddings = nn.Parameter(torch.zeros(num_bins, embed_dim))
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.W_q.weight)
        nn.init.eye_(self.W_k.weight)
        self.W_q.weight.data.mul_(0.01)
        self.W_k.weight.data.mul_(0.01)

        self.register_buffer("raw_counts", torch.ones(num_bins))

    def set_raw_counts(self, labels, reweight="sqrt_inv"):
        counts = compute_raw_bin_counts(labels, self.num_bins, reweight=reweight)
        self.raw_counts.copy_(torch.tensor(counts, dtype=torch.float32))

    def kernel_logits(self):
        e = self.embeddings
        q = self.W_q(e)
        k = self.W_k(e)
        residual = torch.matmul(q, k.t()) - self.lambda_d * self.bin_distance
        if self.unconstrained:
            return residual
        return self.init_logits + self.residual_scale * residual

    def kernel_matrix(self):
        return F.softmax(self.kernel_logits(), dim=1)

    def effective_density(self):
        K = self.kernel_matrix()
        return torch.matmul(K, self.raw_counts)

    def bin_weights(self):
        p = self.effective_density()
        w = 1.0 / torch.pow(p + self.epsilon, self.beta)
        return w * (self.num_bins / w.sum())

    def sample_weights(self, targets):
        """targets: (N, 1) float tensor on same device as module."""
        w_bins = self.bin_weights()
        idx = targets.long().clamp(0, self.num_bins - 1).view(-1)
        return w_bins[idx].view(-1, 1)

    def locality_regularization(self):
        K = self.kernel_matrix()
        return (K * self.bin_distance).sum()

    def smoothness_regularization(self):
        K = self.kernel_matrix()
        return (K[1:] - K[:-1]).pow(2).sum()

    def symmetry_regularization(self):
        K = self.kernel_matrix()
        return (K - K.t()).pow(2).sum()

    def kernel_regularization(self, lam_local, lam_smooth, lam_sym):
        return (
            lam_local * self.locality_regularization()
            + lam_smooth * self.smoothness_regularization()
            + lam_sym * self.symmetry_regularization()
        )


def bin_balanced_meta_loss(per_bin_losses, per_bin_counts, K_smooth, tau=5.0):
    """
    per_bin_losses: dict bin -> tensor scalar loss (with grad or detached)
    per_bin_counts: dict bin -> int count in meta batch
    K_smooth: (B, B) smoothing kernel (K0 buffer or learned K_theta)
    """
    bins = sorted(per_bin_losses.keys())
    if not bins:
        return torch.tensor(0.0)

    device = next(iter(per_bin_losses.values())).device
    B = K_smooth.shape[0]
    loss_vec = torch.zeros(B, device=device)
    count_vec = torch.zeros(B, device=device)
    for b, loss in per_bin_losses.items():
        if 0 <= b < B:
            loss_vec[b] = loss
            count_vec[b] = float(per_bin_counts[b])

    active = count_vec > 0
    if active.sum() == 0:
        return torch.tensor(0.0, device=device)

    rho = count_vec / (count_vec + tau)
    smoothed = torch.matmul(K_smooth, loss_vec)
    bar_loss = rho * loss_vec + (1.0 - rho) * smoothed
    return bar_loss[active].mean()


def accumulate_meta_bin_losses(model, meta_loader, loss_fn, device, max_batches=None):
    """One forward pass over meta loader; return per-bin mean losses and counts."""
    model.eval()
    bin_sum = {}
    bin_cnt = {}
    n_batches = 0
    for inputs, targets, _ in meta_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs, targets, 0) if hasattr(model, "module") else model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        per_sample = F.l1_loss(outputs, targets, reduction="none")
        for i in range(targets.size(0)):
            b = int(targets[i].item())
            if b not in bin_sum:
                bin_sum[b] = per_sample[i].detach()
                bin_cnt[b] = 1
            else:
                bin_sum[b] = bin_sum[b] + per_sample[i].detach()
                bin_cnt[b] += 1
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    per_bin_losses = {b: bin_sum[b] / bin_cnt[b] for b in bin_sum}
    return per_bin_losses, bin_cnt
