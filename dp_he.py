import math
import warnings

import numpy as np
import torch

try:
    from Pyfhel import Pyfhel
    _HAS_PYFHEL = True
except Exception:
    Pyfhel = None
    _HAS_PYFHEL = False


DP_CLIP_NORM = 0.5
DP_NOISE_MULTIPLIER = 1.5
DP_ADAPTIVE_CLIPPING = True
DP_CLIP_PERCENTILE = 30
DP_EXPOSE_GRAD_NORM = False  # set True only for debugging


class AdaptiveClipper:
    def __init__(self, initial_norm=1.0, percentile=50, ema=0.9):
        self.current_norm = float(initial_norm)
        self.percentile = float(percentile)
        self.ema = float(ema)
        self.norm_history = []

    def update(self, gradient_norms):
        self.norm_history.extend([float(x) for x in gradient_norms])
        if len(self.norm_history) > 200:
            self.norm_history = self.norm_history[-200:]
        if len(self.norm_history) >= 20:
            new_norm = float(np.percentile(self.norm_history, self.percentile))
            self.current_norm = self.ema * self.current_norm + (1.0 - self.ema) * new_norm

    def get_norm(self):
        return max(float(self.current_norm), 0.05)


ADAPTIVE_CLIPPER = AdaptiveClipper(DP_CLIP_NORM, DP_CLIP_PERCENTILE)


def dp_clip_and_noise_flat(update, C=None, sigma=None, generator=None, adaptive=True):
    """L2-clip an update (list of tensors) and add Gaussian noise.

    Client-level DP: clip each client's whole-update to norm C and add
    Gaussian noise with std = sigma * C elementwise.
    Returns (noised_update_list, grad_norm).
    """

    if C is None:
        C = DP_CLIP_NORM
    if sigma is None:
        sigma = DP_NOISE_MULTIPLIER

    flat = torch.cat([u.detach().view(-1) for u in update])
    nrm = float(flat.norm(p=2).item())

    if adaptive and DP_ADAPTIVE_CLIPPING:
        ADAPTIVE_CLIPPER.update([nrm])
        C = ADAPTIVE_CLIPPER.get_norm()

    if nrm > C:
        flat = flat * (C / (nrm + 1e-10))

    if generator is None:
        noise = torch.randn_like(flat)
    else:
        noise = torch.randn_like(flat, generator=generator)
    flat = flat + noise * (sigma * C)

    out, offset = [], 0
    for u in update:
        numel = u.numel()
        chunk = flat[offset:offset + numel].reshape(u.shape)
        out.append(chunk.to(u.device).type_as(u))
        offset += numel

    if DP_EXPOSE_GRAD_NORM:
        return out, nrm
    return out, nrm


def compute_epsilon_from_sigma_zcdp(sigma, C=None, steps=1, delta=1e-5):
    """Approximate epsilon from Gaussian noise via zCDP conversion.

    rho = steps * C^2 / (2 * sigma^2)
    epsilon = rho + 2 * sqrt(rho * log(1/delta))
    """
    if C is None:
        C = DP_CLIP_NORM
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    rho = float(steps) * (float(C) ** 2) / (2.0 * (sigma ** 2))
    if rho <= 0.0:
        return 0.0
    eps = rho + 2.0 * math.sqrt(rho * math.log(1.0 / float(delta)))
    return float(eps)


def compute_epsilon_from_sigma_rdp(sigma, steps=1, delta=1e-5, orders=None, sampling_rate=None):
    """Estimate (epsilon, optimal_order) from `sigma` using RDP (central Gaussian).

    This is an approximation. For production use a dedicated accountant.
    """
    if orders is None:
        orders = list(range(2, 65))
    orders = np.array(orders, dtype=float)

    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    eff_steps = float(steps)
    if (sampling_rate is not None) and (sampling_rate > 0.0) and (sampling_rate < 1.0):
        warnings.warn("sampling_rate < 1 provided: using crude scaling of steps. "
                      "Use a proper subsampled RDP accountant for accuracy.")
        eff_steps = eff_steps * float(sampling_rate)

    rdp = eff_steps * (orders) / (2.0 * (sigma ** 2))
    epsilons = rdp + (np.log(1.0 / float(delta)) / (orders - 1.0))
    idx = int(np.argmin(epsilons))
    return float(epsilons[idx]), float(orders[idx])


def sigma_from_epsilon_rdp(epsilon, steps=1, delta=1e-5, orders=None, sampling_rate=None):
    """Compute a conservative sigma such that the central Gaussian mechanism
    approximately achieves `epsilon` after `steps`.
    """
    if orders is None:
        orders = list(range(2, 65))
    orders = np.array(orders, dtype=float)

    eff_steps = float(steps)
    if (sampling_rate is not None) and (sampling_rate > 0.0) and (sampling_rate < 1.0):
        warnings.warn("sampling_rate < 1 provided: using crude scaling of steps. "
                      "Use a proper subsampled RDP accountant for accuracy.")
        eff_steps = eff_steps * float(sampling_rate)

    best_sigma = None
    for alpha in orders:
        denom = (epsilon - (np.log(1.0 / float(delta)) / (alpha - 1.0)))
        if denom <= 0:
            continue
        rdp_needed = denom
        sigma_alpha = math.sqrt((eff_steps * alpha) / (2.0 * rdp_needed))
        if (best_sigma is None) or (sigma_alpha < best_sigma):
            best_sigma = sigma_alpha

    if best_sigma is None:
        raise ValueError("Could not find sigma for given epsilon/delta with chosen orders.")
    return float(best_sigma)


# -------------------- HE configuration and helpers --------------------
HE = None
HE_PARAMS = {
    "scheme": "CKKS",
    "n": 2 ** 15,
    "scale": 2 ** 40,
    "qi_sizes": [60, 40, 40, 40, 60],
}


def init_he_context():
    global HE
    if HE is not None:
        return
    if not _HAS_PYFHEL:
        raise RuntimeError("Pyfhel is not available, cannot use HE helpers.")
    he = Pyfhel()
    he.contextGen(**HE_PARAMS)
    he.keyGen()
    try:
        he.rotateKeyGen()
        he.relinKeyGen()
    except Exception:
        pass
    HE = he


def he_encrypt_tensor(t: torch.Tensor, chunk=4096):
    init_he_context()
    x = t.detach().cpu().view(-1).numpy().astype(np.float64)

    cts = []
    sizes = []

    try:
        max_slots = int(HE_PARAMS.get("n", 1) // 2)
    except Exception:
        max_slots = None
    if (max_slots is not None) and (chunk > max_slots):
        warnings.warn(f"HE chunk ({chunk}) > available CKKS slots ({max_slots}). This may fail or be suboptimal.")

    for i in range(0, len(x), chunk):
        block = x[i:i + chunk]
        sizes.append(len(block))
        pt = HE.encodeFrac(block)
        ct = HE.encryptPtxt(pt)
        cts.append(ct)

    return cts, t.shape, sizes


def he_sum_ciphertexts(ct_lists):
    m = len(ct_lists[0])
    out = []
    for j in range(m):
        acc = ct_lists[0][j].copy()
        for k in range(1, len(ct_lists)):
            acc += ct_lists[k][j]
        out.append(acc)
    return out


def he_decrypt_tensor(cts, shape, sizes):
    init_he_context()
    parts = []
    for ct, size in zip(cts, sizes):
        arr = np.array(HE.decryptFrac(ct))
        arr = arr[:size]
        parts.append(arr)
    flat = np.concatenate(parts, axis=0)
    return torch.from_numpy(flat.reshape(shape))


def he_export_public_parameters():
    init_he_context()
    try:
        ctx_bytes = HE.to_bytes_context()
        pk_bytes = HE.to_bytes_public_key()
        return {"context": ctx_bytes, "public_key": pk_bytes}
    except Exception:
        raise RuntimeError("Pyfhel serialization not available in this environment.")


def he_note_key_management():
    warnings.warn(
        "HE helper: current `init_he_context()` generates secret keys in-process. "
        "For secure deployments do NOT keep secret keys on an untrusted server. "
        "Consider threshold-HE or server-side key custody."
    )
