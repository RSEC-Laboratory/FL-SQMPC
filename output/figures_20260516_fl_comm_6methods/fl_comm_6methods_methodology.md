# FL-SQMPC per-round communication cost - methodology

Companion to ``fl_comm_6methods_*`` figures. Every bar in those figures is an analytical estimate of bytes exchanged in **one** federated round. Rounds-to-completion (50/100/150) are deliberately omitted so the per-round structure of each protocol is exposed.

## Federation parameters

- N_clients = **5** participating edges per round
- n_agg = **2** SMPC aggregators (``sqmpc_core.SMPC_NUM_AGGREGATORS``)
- batch size b = **100** (matches ``run_fl_sqmpc_accuracy.py`` / ``run_cidiot_accuracy.py`` defaults)
- float32 wire format = 4 bytes per scalar
- FL-SQMPC quantisation = **2 bits/param across all 3 layers**, share width = ceil(log2(5 * 2^2)) = **5 bits/share**

## Model and dataset dimensions

All non-CIDIoT methods aggregate the same MLP ``[d -> 128 -> 64 -> C]`` with biases.

| Dataset | d (features) | C (classes) | P (params) | |G| (CIDIoT generator) |
|---|---|---|---|---|
| BotIoT Multiclass | 65 | 4 | 16,964 | 256,129 |
| CICIoT2023 Multiclass | 39 | 8 | 13,896 | 158,849 |
| ToN-IoT Multiclass | 39 | 10 | 14,026 | 158,881 |

## Per-method formulas

### Vanilla FL

- **Client per round:** ``4 * P``
- **Server per round:** ``N_clients * 4 * P (both directions)``
- **Source:** analytical
- **Notes:** FedAvg with float32 model weights.

### HE

- **Client per round:** ``ceil(P / slots) * ct_size``
- **Server per round:** ``N_clients * ceil(P / slots) * ct_size``
- **Source:** analytical; CKKS N_poly=8192, ct approx 440 KB, slots=4096
- **Notes:** Microsoft SEAL default 128-bit security parameters.

### SMPC

- **Client per round:** ``n_agg * 4 * P (up), 4 * P (dn)``
- **Server per round:** ``N_clients * n_agg * 4 * P (in), N_clients * 4 * P (out)``
- **Source:** analytical; n_agg=2
- **Notes:** Full-width additive secret shares to each aggregator.

### DP

- **Client per round:** ``4 * P``
- **Server per round:** ``same as Vanilla FL``
- **Source:** analytical
- **Notes:** DP noise added pre-transmission; wire format unchanged.

### CIDIoT

- **Client per round:** ``b * |x|``
- **Server per round:** ``N_clients * b * |x|``
- **Source:** Yao et al. IoT-J 2024, Sec. VI.B & Table VI (recomputed on our feature dims)
- **Notes:** Paper-reported whole-network 180/169 KB on CIC_IoT2023/ToN_IoT; transmits batch data flows + sample gradients, not generator weights.

### FL-SQMPC

- **Client per round:** ``n_agg * ceil(log2(N * 2^q)) / 8 * P (up), 4 * P (dn)  [q=2, share=5 bits]``
- **Server per round:** ``N_clients * n_agg * ceil(log2(N * 2^q)) / 8 * P (in), N_clients * 4 * P (out)``
- **Source:** analytical; q=2 bits/param across all 3 layers, additive secret-sharing across 2 aggregators
- **Notes:** Quantised weights are additively secret-shared. The share ring must satisfy M >= N_clients * 2^q so the sum of shares recovers the correct aggregate without modular wrap; this sets the wire width per share.

## CIDIoT in detail

Yao et al. (IEEE IoT-J vol. 11 no. 9, May 2024) state in Section VI.B that the per-round communication cost of CIDIoT is ``O(2 K b |x| + K l_ss)``, where ``K`` is the number of edges, ``b`` the batch size, ``|x|`` the size of one data flow (one input sample at float32 width), and ``l_ss`` the secret-share overhead (negligible vs. ``|x|``). Their Table VI reports whole-network per-round comm:

- CIC_IoT2023: **180 KB / round**
- ToN_IoT: **169 KB / round**

With their setup (K=5, b=100, |x|=47x4 B for CIC_IoT2023) the formula gives 2 x 5 x 100 x 188 B = 188 KB, matching the reported 180 KB within rounding. We recompute this formula on **our** processed feature dims (d=65/39/39 for Bot-IoT/CIC/ToN) so CIDIoT is compared like-for-like with the other methods. Bot-IoT is not covered by the paper; we extrapolate via 2 K b |x|.

**Generator-weight alternative.** If CIDIoT were implemented by federating the conditional TCN generator weights instead, the per-client cost would be ``4 |G|`` each way. We record that variant in the CSV (``method = 'CIDIoT (gen weights)'``) for completeness, but it is **not** what the paper does and is **not** the primary CIDIoT bar.

## FL-SQMPC in detail

Each client quantises its weight update to **q = 2 bits/param** on all three Linear layers (signed symmetric quantisation, see ``sqmpc_core._quantize_dequantize_symmetric``). The quantised integer value is then **additively secret-shared** across n_agg = 2 aggregators: shares ``r_1, ..., r_{n_agg}`` are drawn from a ring ``Z_M`` with ``sum_i r_i = quantised_value (mod M)``.

For the aggregator to recover the correct sum across N_clients = 5 clients without modular wrap, the ring must satisfy ``M >= N_clients * 2^q`` = ``5 * 2^2`` = ``20``. Hence each share occupies ``ceil(log2(5 * 2^2)) = 5 bits`` on the wire, and the **per-client upload** is ``n_agg * 5 * P / 8`` bytes per round - i.e. ``1.250 * P`` bytes/round.

The download (server -> client broadcast of the aggregated model) is left at full float32 precision, ``4 * P`` bytes. Compressing the broadcast would require an additional protocol step (e.g. quantised broadcast) and is out of scope for this figure.

**Why the share width depends on N_clients.** Naively one might say "q=2 bits times n_agg=2 shares = 4 bits/param." That under-counts the secret-sharing overhead: each individual share must be uniformly distributed over a ring large enough that the **aggregate** of all N_clients shares does not wrap. Increasing N_clients increases the wire width per share logarithmically.

**Accuracy at q=2** is characterised in ``output/figures_20260512_sqmpc_bits_ablation/`` - the precision drop vs full-float aggregation is small (the figure's narrative argument).

## Per-dataset bar values

### BotIoT Multiclass

**client per round:**

| Method | upload | download | total |
|---|---|---|---|
| Vanilla FL | 66.3 KB | 66.3 KB | 132.5 KB |
| HE | 2.1 MB | 2.1 MB | 4.3 MB |
| SMPC | 132.5 KB | 66.3 KB | 198.8 KB |
| DP | 66.3 KB | 66.3 KB | 132.5 KB |
| CIDIoT | 25.4 KB | 25.4 KB | 50.8 KB |
| FL-SQMPC | 20.7 KB | 66.3 KB | 87.0 KB |

**server per round:**

| Method | upload | download | total |
|---|---|---|---|
| Vanilla FL | 331.3 KB | 331.3 KB | 662.7 KB |
| HE | 10.7 MB | 10.7 MB | 21.5 MB |
| SMPC | 662.7 KB | 331.3 KB | 994.0 KB |
| DP | 331.3 KB | 331.3 KB | 662.7 KB |
| CIDIoT | 127.0 KB | 127.0 KB | 253.9 KB |
| FL-SQMPC | 103.5 KB | 331.3 KB | 434.9 KB |

**Recorded alternatives (not plotted):**

- CIDIoT (gen weights) (client): 2.0 MB - analytical (this work); if generator weights were federated
- CIDIoT (gen weights) (server): 9.8 MB - analytical (this work)

### CICIoT2023 Multiclass

**client per round:**

| Method | upload | download | total |
|---|---|---|---|
| Vanilla FL | 54.3 KB | 54.3 KB | 108.6 KB |
| HE | 1.7 MB | 1.7 MB | 3.4 MB |
| SMPC | 108.6 KB | 54.3 KB | 162.8 KB |
| DP | 54.3 KB | 54.3 KB | 108.6 KB |
| CIDIoT | 15.2 KB | 15.2 KB | 30.5 KB |
| FL-SQMPC | 17.0 KB | 54.3 KB | 71.2 KB |

**server per round:**

| Method | upload | download | total |
|---|---|---|---|
| Vanilla FL | 271.4 KB | 271.4 KB | 542.8 KB |
| HE | 8.6 MB | 8.6 MB | 17.2 MB |
| SMPC | 542.8 KB | 271.4 KB | 814.2 KB |
| DP | 271.4 KB | 271.4 KB | 542.8 KB |
| CIDIoT | 76.2 KB | 76.2 KB | 152.3 KB |
| FL-SQMPC | 84.8 KB | 271.4 KB | 356.2 KB |

**Recorded alternatives (not plotted):**

- CIDIoT (gen weights) (client): 1.2 MB - analytical (this work); if generator weights were federated
- CIDIoT (gen weights) (server): 6.1 MB - analytical (this work)
- CIDIoT (paper Table VI) (server): 180.0 KB - Yao et al. IoT-J 2024, Table VI

### ToN-IoT Multiclass

**client per round:**

| Method | upload | download | total |
|---|---|---|---|
| Vanilla FL | 54.8 KB | 54.8 KB | 109.6 KB |
| HE | 1.7 MB | 1.7 MB | 3.4 MB |
| SMPC | 109.6 KB | 54.8 KB | 164.4 KB |
| DP | 54.8 KB | 54.8 KB | 109.6 KB |
| CIDIoT | 15.2 KB | 15.2 KB | 30.5 KB |
| FL-SQMPC | 17.1 KB | 54.8 KB | 71.9 KB |

**server per round:**

| Method | upload | download | total |
|---|---|---|---|
| Vanilla FL | 273.9 KB | 273.9 KB | 547.9 KB |
| HE | 8.6 MB | 8.6 MB | 17.2 MB |
| SMPC | 547.9 KB | 273.9 KB | 821.8 KB |
| DP | 273.9 KB | 273.9 KB | 547.9 KB |
| CIDIoT | 76.2 KB | 76.2 KB | 152.3 KB |
| FL-SQMPC | 85.6 KB | 273.9 KB | 359.6 KB |

**Recorded alternatives (not plotted):**

- CIDIoT (gen weights) (client): 1.2 MB - analytical (this work); if generator weights were federated
- CIDIoT (gen weights) (server): 6.1 MB - analytical (this work)
- CIDIoT (paper Table VI) (server): 169.0 KB - Yao et al. IoT-J 2024, Table VI
