"""PRIME: proxy-based representation learning for imbalanced regression (Lim et al., ICML 2025)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_target_dist(y_proxy):
    return (y_proxy.unsqueeze(0) - y_proxy.unsqueeze(1)).abs()


def _cosine_dist(a, b):
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return 1.0 - (a_n * b_n).sum(dim=-1)


def _softmax_rows(logits, zero_diag=True):
    if zero_diag:
        logits = logits.clone()
        logits.fill_diagonal_(-1e9)
    return F.softmax(logits, dim=-1)


def target_similarity(y, y_proxy, tau_t):
    """T vector per sample, shape [B, C]."""
    d = (y.unsqueeze(1) - y_proxy.unsqueeze(0)).abs()
    logits = -tau_t * d
    return F.softmax(logits, dim=-1)


def feature_association(z, z_proxy, tau_f):
    """A vector per sample, shape [B, C]."""
    d = _cosine_dist(z.unsqueeze(1), z_proxy.unsqueeze(0))
    logits = -tau_f * d
    return F.softmax(logits, dim=-1)


def proxy_loss(z_proxy, y_proxy, tau_t, tau_f, alpha, eps=1e-8):
    """L_proxy from PRIME eq. (4)."""
    dt = _pairwise_target_dist(y_proxy)
    df = _cosine_dist(z_proxy.unsqueeze(0), z_proxy.unsqueeze(1))

    logits_p = -tau_t * dt
    p = _softmax_rows(logits_p, zero_diag=True)

    logits_q = -tau_f * df
    q = _softmax_rows(logits_q, zero_diag=True)

    kl = (p * (p.add(eps).log() - q.add(eps).log())).sum()
    reg = (alpha * dt * (1.0 - (1.0 - df).clamp(min=0.0))).sum()
    return kl + reg


def alignment_loss(y, z, z_proxy, y_proxy, tau_t, tau_f, prw=False, delta_min=0.05):
    """L_align or L_align-PRW (eq. 7 / 9)."""
    t = target_similarity(y, y_proxy, tau_t)
    a = feature_association(z, z_proxy, tau_f)
    if prw:
        c = y_proxy.numel()
        b = y.size(0)
        s = (c / b) * t.sum(dim=0)
        smed = s.median().clamp(min=1e-6)
        s_hat = torch.clamp(s, min=delta_min * smed)
        weights = 1.0 / s_hat
        loss = -(weights.unsqueeze(0) * t * a.add(1e-8).log()).sum(dim=-1)
    else:
        loss = -(t * a.add(1e-8).log()).sum(dim=-1)
    return loss.mean()


class ProxyBank(nn.Module):
    """Learnable proxy embeddings with fixed uniformly spaced targets."""

    def __init__(self, num_proxies, feat_dim, y_min, y_max):
        super().__init__()
        self.num_proxies = num_proxies
        y_proxy = torch.linspace(float(y_min), float(y_max), num_proxies)
        self.register_buffer("y_proxy", y_proxy)
        self.z_proxy = nn.Parameter(torch.empty(num_proxies, feat_dim))
        nn.init.kaiming_uniform_(self.z_proxy, a=5 ** 0.5)

    def forward(self):
        return self.z_proxy, self.y_proxy
