"""Batched LIF (Shiu et al. 2024 update rule) on the hearing subnetwork, event-driven synaptic delivery."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import torch

from data import DATA, ROOT, load_config
from ear import audio_hash, jo_drive, stimulus


@dataclass
class Net:
    sex: str
    dataset: str
    n: int
    indptr: torch.Tensor
    indices: torch.Tensor
    data: torch.Tensor
    out_deg: torch.Tensor
    jo_idx: np.ndarray
    jo_subtypes: list[str]
    stage: np.ndarray
    target: np.ndarray
    variant: str
    w_syn: float


def load_subnet(sex: str) -> dict:
    z = np.load(DATA / f"subnet_{sex}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def build_net(sex: str, cfg: dict, sub: dict | None = None, min_weight: int | None = None,
              shuffle_seed: int | None = None, flip_low_conf: bool = False, device: str = "cpu") -> Net:
    sub = sub or load_subnet(sex)
    ds = str(sub["dataset"])
    w_syn = float(cfg["sim"]["w_syn_mv"][ds])
    pre, post, w = sub["pre"].astype(np.int64), sub["post"].astype(np.int64), sub["weight"].astype(np.float64)
    keep = w >= (min_weight or cfg["subnet"]["min_weight"])
    pre, post, w = pre[keep], post[keep], w[keep]
    sign = sub["sign"].astype(np.float64).copy()
    variant = []
    if min_weight and min_weight != cfg["subnet"]["min_weight"]:
        variant.append(f"w>={min_weight}")
    if flip_low_conf:
        low = sub["nt_conf"] < cfg["signs"]["low_conf_threshold"]
        sign[low] *= -1
        variant.append(f"flip{int(low.sum())}")
    if shuffle_seed is not None:
        # Degree-preserving: permute targets across edges (keeps every in- and out-degree and weight multiset).
        post = np.random.default_rng(shuffle_seed).permutation(post)
        variant.append("shuffled")
    n = len(sub["ids"])
    vals = w_syn * sign[pre] * w
    order = np.lexsort((post, pre))
    pre, post, vals = pre[order], post[order], vals[order]
    nz = vals != 0
    pre, post, vals = pre[nz], post[nz], vals[nz]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    dev = torch.device(device)
    jo = np.flatnonzero(sub["stage"] == "jo")
    target = sub["is_song"] | sub["is_recep"]
    return Net(sex, ds, n, torch.as_tensor(indptr, device=dev), torch.as_tensor(post, device=dev),
               torch.as_tensor(vals, dtype=torch.float32, device=dev),
               torch.as_tensor(np.diff(indptr), device=dev), jo, [str(s) for s in sub["jo_subtype"][jo]],
               sub["stage"], target, "+".join(variant) or "baseline", w_syn)


def jo_events(drive: np.ndarray, rate_max: float, base: float, dt_ms: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Inhomogeneous Poisson kicks: rate = base + rate_max * drive. Returns (step, jo_local) sorted by step."""
    rng = np.random.default_rng(seed)
    steps, nids = [], []
    chunk = 2000
    for t0 in range(0, drive.shape[1], chunk):
        p = (base + rate_max * drive[:, t0:t0 + chunk]) * dt_ms / 1000.0
        i, t = np.nonzero(rng.random(p.shape, dtype=np.float32) < p)
        steps.append(t + t0)
        nids.append(i)
    s, i = np.concatenate(steps), np.concatenate(nids)
    o = np.argsort(s, kind="stable")
    return s[o], i[o]


def simulate(net: Net, audios: list[np.ndarray], seeds: list[int], cfg: dict, record: bool = False,
             silence: np.ndarray | None = None, bin_ms: float = 10.0) -> dict:
    """audios[k]: waveform at ear fs for trial k (seed k drives ear jitter + Poisson draws).
    Returns counts (B, n), stage rates (B, S, bins), optional spikes."""
    sc = cfg["sim"]
    dt = sc["dt_ms"]
    B, T, n = len(audios), min(len(a) for a in audios), net.n
    dev = net.data.device
    a_v, b_g = math.exp(-dt / sc["tau_membrane_ms"]), math.exp(-dt / sc["tau_synapse_ms"])
    tm, ts = sc["tau_membrane_ms"], sc["tau_synapse_ms"]
    c_gv = ts / (ts - tm) * (b_g - a_v)
    v0term = sc["v_rest_mv"] * (1 - a_v)
    v_th, v_reset = sc["v_threshold_mv"], sc["v_reset_mv"]
    delay = round(sc["delay_ms"] / dt)
    refr = round(sc["t_refractory_ms"] / dt)
    kick = net.w_syn * 250.0
    data = net.data
    if silence is not None and len(silence):
        m = torch.ones(n, device=dev)
        m[torch.as_tensor(silence, device=dev)] = 0
        pre_of = torch.repeat_interleave(torch.arange(n, device=dev), net.out_deg)
        data = data * m[pre_of]
    # Poisson kicks for every trial, flattened to keys trial*n + neuron, grouped by step.
    ks, kk = [], []
    jo = torch.as_tensor(net.jo_idx, device=dev)
    for b, (a, sd) in enumerate(zip(audios, seeds)):
        s, i = jo_events(drive_for(net, a, cfg, sd)[:, :T], sc["jo_max_rate_hz"], sc["jo_baseline_hz"], dt, sd)
        ks.append(s)
        kk.append(b * n + net.jo_idx[i])
    kstep = np.concatenate(ks)
    o = np.argsort(kstep, kind="stable")
    kstep, kkey = kstep[o], torch.as_tensor(np.concatenate(kk)[o], device=dev)
    koff = np.searchsorted(kstep, np.arange(T + 1))
    reload = torch.full((B * n,), refr, dtype=torch.int16, device=dev)
    reload[(torch.arange(B, device=dev)[:, None] * n + jo[None, :]).ravel()] = 0  # driven neurons: no refractory
    v = torch.full((B * n,), sc["v_rest_mv"], device=dev)
    g = torch.zeros(B * n, device=dev)
    rfc = torch.zeros(B * n, dtype=torch.int16, device=dev)
    ring = torch.zeros((delay + 1, B * n), device=dev)
    counts = torch.zeros(B * n, dtype=torch.int32, device=dev)
    stages = list(cfg["subnet"]["stage_order"])
    st_idx = torch.as_tensor(np.array([stages.index(s) for s in net.stage]), device=dev)
    st_n = np.bincount(np.array([stages.index(s) for s in net.stage]), minlength=len(stages)).clip(1)
    bin_steps = int(round(bin_ms / dt))
    nb = T // bin_steps
    st_counts = torch.zeros((B, len(stages), nb + 1), device=dev)
    rec_t, rec_k = [], []
    for t in range(T):
        slot = t % (delay + 1)
        syn = ring[slot]
        r = (rfc - 1).clamp_(min=0)
        nr = r == 0
        vn = torch.where(nr, v0term + g * c_gv + v * a_v, v)
        gn = torch.where(nr, g * b_g + syn, g)
        sp = nr & (vn > v_th)
        v = torch.where(sp, v_reset, vn)
        g = torch.where(sp, 0.0, gn)
        rfc = torch.where(sp, reload, r)
        syn.zero_()
        lo, hi = koff[t], koff[t + 1]
        if hi > lo:
            k = kkey[lo:hi]
            ok = (rfc[k] == 0) & ~sp[k]
            v.index_add_(0, k, torch.where(ok, kick, 0.0))
        idx = sp.nonzero().squeeze(1)
        if idx.numel():
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
            trial, pre = idx // n, idx % n
            st_counts.index_put_((trial, st_idx[pre], torch.full_like(trial, t // bin_steps)),
                                 torch.ones(idx.numel(), device=dev), accumulate=True)
            if record:
                rec_t.append(np.full(idx.numel(), t, dtype=np.int32))
                rec_k.append(idx.cpu().numpy())
            deg = net.out_deg[pre]
            tot = int(deg.sum())
            if tot:
                cum = torch.cumsum(deg, 0) - deg
                within = torch.arange(tot, device=dev) - torch.repeat_interleave(cum, deg, output_size=tot)
                e = torch.repeat_interleave(net.indptr[pre], deg, output_size=tot) + within
                keys = torch.repeat_interleave(trial, deg, output_size=tot) * n + net.indices[e]
                ring[(t + delay) % (delay + 1)].index_add_(0, keys, data[e])
    # Note: an index_add_ with duplicate keys sums duplicates, as required.
    dur_s = T * dt / 1000.0
    out = {
        "counts": counts.view(B, n).cpu().numpy(),
        "rates": counts.view(B, n).cpu().numpy() / dur_s,
        "stage_rates": st_counts[:, :, :nb].cpu().numpy() / st_n[None, :, None] / (bin_ms / 1000.0),
        "stages": stages, "bin_ms": bin_ms, "dur_s": dur_s,
    }
    if record:
        tt = np.concatenate(rec_t) if rec_t else np.zeros(0, np.int32)
        kk_ = np.concatenate(rec_k) if rec_k else np.zeros(0, np.int64)
        out["spikes"] = [((kk_[tt_mask] % n).astype(np.uint32), (tt[tt_mask] * dt / 1000.0).astype(np.float32))
                         for tt_mask in [(kk_ // n) == b for b in range(B)]]
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "uncommitted"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run_meta(net: Net, cfg: dict, audio: np.ndarray, seed: int, **extra) -> dict:
    return {"input_hash": audio_hash(audio), "dataset": net.dataset, "sex": net.sex, "variant": net.variant,
            "config_hash": config_hash(cfg), "seed": int(seed), "git_commit": git_commit(),
            "w_syn_mv": net.w_syn, "n_neurons": net.n, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}


def drive_for(net: Net, audio: np.ndarray, cfg: dict, seed: int) -> np.ndarray:
    return jo_drive(audio, cfg["ear"]["fs"], net.jo_subtypes, cfg["ear"], seed=seed)


if __name__ == "__main__":
    cfg = load_config()
    torch.set_num_threads(4)
    audio = stimulus("pulse", cfg)
    for sex in ("male", "female"):
        net = build_net(sex, cfg)
        t0 = time.time()
        r = simulate(net, [audio], [0], cfg, record=True)
        el = time.time() - t0
        st = {s: int(r["counts"][0][net.stage == s].sum()) for s in r["stages"]}
        print(f"{sex}: n={net.n} 1 s pulse song in {el:.1f}s; spikes per stage {st}; "
              f"target mean rate {r['rates'][0][net.target].mean():.2f} Hz")
