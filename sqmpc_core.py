import math
import numpy as np
import torch

SMPC_NUM_AGGREGATORS = 3
SMPC_MODULUS = 2**31 - 1
SMPC_SCALE = 1e12
SMPC_MAX_CLIENTS = 100
SMPC_CLIP_VALUE = 0.12
SMPC_SAFETY_MARGIN = 0.9

SQMPC_Q_BITS_FIRST = 14
SQMPC_Q_BITS_MID = 14
SQMPC_Q_BITS_LAST = 12
SQMPC_Q_CLIP_PERCENTILE = 90.0
SQMPC_Q_EPS = 1e-8
SQMPC_USE_DYNAMIC_CLIENT_CLIP = False
SQMPC_Q_SCOPE = "all"
# When True and bits==1, zero raw-update elements are promoted to +scale
# (instead of staying at 0). Strictly 2-level sign quantization.
SQMPC_Q_B1_STRICT = False


def stochastic_round(x):
    floor = torch.floor(x)
    prob = x - floor
    return floor + (torch.rand_like(x) < prob).to(x.dtype)


def _percentile_scale(u, percentile, eps):
    u_abs = u.detach().abs().view(-1)
    if u_abs.numel() == 0:
        return eps
    p = float(percentile) / 100.0
    try:
        q = torch.quantile(u_abs, p)
        scale = float(q.item())
    except Exception:
        scale = float(np.percentile(u_abs.detach().cpu().numpy(), percentile))
    return max(scale, eps)


def _quantize_dequantize_symmetric(u, bits):
    if bits < 1:
        raise ValueError("bits must be >= 1")

    scale = _percentile_scale(u, SQMPC_Q_CLIP_PERCENTILE, SQMPC_Q_EPS)

    if bits == 1:
        # Sign-only quantization, SignSGD-style.
        # Default: torch.sign maps 0 -> 0, so exactly-zero raw-update elements
        # remain 0 in the output -- effectively 3 levels {-scale, 0, +scale}.
        # When SQMPC_Q_B1_STRICT is True, force everything to +/-scale
        # (strictly 2-level output).
        v = torch.clamp(u / scale, -1.0, 1.0)
        if SQMPC_Q_B1_STRICT:
            return torch.where(v >= 0, torch.full_like(v, scale), torch.full_like(v, -scale))
        return torch.sign(v) * scale

    qmax = float((1 << (bits - 1)) - 1)
    v = torch.clamp(u / scale, -1.0, 1.0)
    q_float = v * qmax
    q_int = stochastic_round(q_float)
    q_int = torch.clamp(q_int, -qmax, qmax)

    v_hat = q_int / qmax
    return v_hat * scale


def mixed_quantize_mlp(updates):
    out = []
    last_idx = len(updates) - 1
    for layer_id, u in enumerate(updates):
        if SQMPC_Q_SCOPE == "none":
            out.append(u)
            continue
        if SQMPC_Q_SCOPE == "last" and layer_id != last_idx:
            out.append(u)
            continue
        if SQMPC_Q_SCOPE == "first" and layer_id != 0:
            out.append(u)
            continue

        if layer_id == 0:
            bits = SQMPC_Q_BITS_FIRST
        elif layer_id == last_idx:
            bits = SQMPC_Q_BITS_LAST
        else:
            bits = SQMPC_Q_BITS_MID
        out.append(_quantize_dequantize_symmetric(u, bits))
    return out


def _encode_to_int_scaled(u, scale):
    x = torch.round(u * scale).to(torch.int32)
    return x.remainder(SMPC_MODULUS)


def _decode_from_int_scaled(x, device, dtype, scale):
    x = x.remainder(SMPC_MODULUS)
    half = SMPC_MODULUS // 2
    x = torch.where(x >= half, x - SMPC_MODULUS, x).to(torch.float64)
    x = x / scale
    return x.to(device).type(dtype)


def smpc_get_layer_scales(num_layers, max_clients=SMPC_MAX_CLIENTS, clip=SMPC_CLIP_VALUE):
    bound = int((SMPC_MODULUS // 2 - 1) * SMPC_SAFETY_MARGIN)
    per_client_bound = max(1, bound // int(max_clients))

    def scale_for_clip(c):
        return float(min(SMPC_SCALE, per_client_bound / (float(c) + 1e-12)))

    if isinstance(clip, (list, tuple)):
        if len(clip) != num_layers:
            raise ValueError("clip list length must match number of layers")
        return [scale_for_clip(c) for c in clip]
    return [scale_for_clip(clip) for _ in range(num_layers)]


def smpc_split(update, num_aggregators=SMPC_NUM_AGGREGATORS, scales=None, clip=None):
    if scales is None:
        raise ValueError("SQmpc requires explicit per-layer scales")

    shares = [[] for _ in range(num_aggregators)]

    for layer_idx, u in enumerate(update):
        enc = _encode_to_int_scaled(u, scales[layer_idx])

        r = torch.randint(
            0,
            SMPC_MODULUS,
            size=(num_aggregators - 1, *u.shape),
            dtype=torch.int32,
            device=u.device,
        )
        sum_rand = torch.sum(r, dim=0).remainder(SMPC_MODULUS)
        last_share = (enc - sum_rand).remainder(SMPC_MODULUS)

        for i in range(num_aggregators - 1):
            shares[i].append(r[i])
        shares[-1].append(last_share)

    return shares


def smpc_split_with_scales(update, num_aggregators=SMPC_NUM_AGGREGATORS, max_clients=SMPC_MAX_CLIENTS, clip=None):
    scales = smpc_get_layer_scales(len(update), max_clients=max_clients, clip=SMPC_CLIP_VALUE)
    shares = smpc_split(update, num_aggregators=num_aggregators, scales=scales, clip=None)
    return shares, scales


def smpc_aggregate_sqmpc(shares_list, num_aggregators=SMPC_NUM_AGGREGATORS, scales=None):
    n_clients = len(shares_list)
    if n_clients == 0:
        raise ValueError("Empty shares_list")

    num_layers = len(shares_list[0][0])
    if scales is None or len(scales) != num_layers:
        raise ValueError("SQmpc aggregate requires per-layer scales")

    partial_aggs = [
        [torch.zeros_like(shares_list[0][s][l], dtype=torch.int32) for l in range(num_layers)]
        for s in range(num_aggregators)
    ]

    for k in range(n_clients):
        for s in range(num_aggregators):
            for l in range(num_layers):
                partial_aggs[s][l] = (partial_aggs[s][l] + shares_list[k][s][l].to(torch.int32)).remainder(SMPC_MODULUS)

    global_update = []
    for l in range(num_layers):
        stacked = torch.stack([partial_aggs[s][l] for s in range(num_aggregators)])
        layer_sum = torch.sum(stacked, dim=0).remainder(SMPC_MODULUS)
        ref = shares_list[0][0][l]
        decoded_sum = _decode_from_int_scaled(layer_sum, device=ref.device, dtype=torch.float32, scale=scales[l])
        global_update.append(decoded_sum / float(n_clients))

    return global_update
