"""Export subnetworks, runs and sweep results to web/public/data (binary float32/int32 + JSON)."""
from __future__ import annotations

import json
import shutil

import numpy as np
from scipy import signal

from data import DATA, ROOT, load_config
from ear import band_sos
from sim import build_net, config_hash, load_subnet

OUT = ROOT / "web" / "public" / "data"
RUN_BUDGET_PER_SEX = 62e6  # keeps the whole export under the 150 MB gate


def _norm_positions(xyz: np.ndarray) -> np.ndarray:
    p = xyz.astype(np.float64)
    p[:, 1] *= -1  # EM y axis points ventral; flip so dorsal is up
    c = np.median(p, axis=0)
    p -= c
    s = np.percentile(np.abs(p), 98, axis=0).max()
    return (p / s).astype(np.float32)


def export_sex(sex: str, cfg: dict) -> dict:
    sub = load_subnet(sex)
    net = build_net(sex, cfg, sub)
    d = OUT / sex
    d.mkdir(parents=True, exist_ok=True)
    _norm_positions(sub["xyz"]).tofile(d / "positions.f32")
    ip, ix, w = (net.indptr.numpy().astype(np.int32), net.indices.numpy().astype(np.int32),
                 net.data.numpy().astype(np.float32))
    with open(d / "csr.bin", "wb") as f:
        f.write(ip.tobytes()); f.write(ix.tobytes()); f.write(w.tobytes())
    syn = np.zeros(len(ix), np.int32)  # raw synapse counts per CSR entry for the inspector
    lookup = {(int(a), int(b)): int(c) for a, b, c in zip(sub["pre"], sub["post"], sub["weight"])}
    pre_of = np.repeat(np.arange(len(ip) - 1), np.diff(ip))
    syn[:] = [lookup.get((int(a), int(b)), 0) for a, b in zip(pre_of, ix)]
    syn.tofile(d / "csr_syn.i32")
    types, tidx = np.unique(sub["cell_type"], return_inverse=True)
    stages = cfg["subnet"]["stage_order"]
    meta = {
        "sex": sex, "dataset": str(sub["dataset"]), "n": int(len(sub["ids"])), "nnz": int(len(ix)),
        "ids": [str(i) for i in sub["ids"]], "types": types.tolist(), "type_idx": tidx.astype(int).tolist(),
        "stage": [stages.index(s) for s in sub["stage"]], "hop": sub["hop"].tolist(),
        "nt": sub["nt"].tolist(), "nt_conf": np.round(sub["nt_conf"], 2).tolist(), "sign": sub["sign"].tolist(),
        "target": np.flatnonzero(net.target).tolist(), "song": np.flatnonzero(sub["is_song"]).tolist(),
        "recep": np.flatnonzero(sub["is_recep"]).tolist(), "jo": net.jo_idx.tolist(), "jo_subtype": net.jo_subtypes,
        "pos_measured": sub["pos_measured"].astype(int).tolist(), "w_syn_mv": net.w_syn,
    }
    (d / "neurons.json").write_text(json.dumps(meta, separators=(",", ":")))
    runs = []
    rd = DATA / "runs" / sex
    od = d / "runs"
    if od.exists():
        shutil.rmtree(od)
    od.mkdir()
    files = sorted(rd.glob("*.npz")) if rd.exists() else []
    files.sort(key=lambda p: (p.stem.startswith("scr_"), p.stem.startswith("pulse_"), p.stem))  # refs, pulse grid, scrambled
    used, skipped = 0, 0
    for p in files:
        z = np.load(p)
        size = (2 * len(z["t"]) + len(z["rates"]) + len(z["audio"]) + z["stage_rates"].size) * 4
        if used + size > RUN_BUDGET_PER_SEX:
            skipped += 1
            continue
        used += size
        o = np.argsort(z["t"], kind="stable")
        ids, t = z["ids"][o].astype(np.uint32), z["t"][o].astype(np.float32)
        rates, audio, sr = z["rates"].astype(np.float32), z["audio"].astype(np.float32), z["stage_rates"].astype(np.float32)
        with open(od / f"{p.stem}.bin", "wb") as f:
            for a in (ids, t, rates, audio, sr):
                f.write(a.tobytes())
        m = json.loads(str(z["meta"]))
        runs.append({"id": p.stem, "file": f"{sex}/runs/{p.stem}.bin", "n_spikes": int(len(ids)), "n": meta["n"],
                     "n_audio": int(len(audio)), "stage_rates_shape": list(sr.shape), "meta": m})
    if skipped:
        print(f"  {sex}: {skipped} runs not exported (size budget {RUN_BUDGET_PER_SEX / 1e6:.0f} MB)")
    return {"sex": sex, "dataset": meta["dataset"], "n": meta["n"], "nnz": meta["nnz"], "runs": runs}


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    sexes = {s: export_sex(s, cfg) for s in ("male", "female")}
    res_p = DATA / "results" / "results.json"
    if res_p.exists():
        r = json.loads(res_p.read_text())
        r.pop("raw", None)
        (OUT / "sweep.json").write_text(json.dumps(r, separators=(",", ":")))
        shutil.copy(DATA / "results" / "heatmaps.png", OUT / "heatmaps.png")
    sc, ec = cfg["subnet"], cfg["ear"]
    scales = 1.0 + ec["jitter_cf_sd"] * np.array([-1.0, 0.0, 1.0])
    ear_sos = {k: [band_sos(b, ec["fs"], s).tolist() for s in scales] for k, b in ec["bands"].items()}
    env_sos = signal.butter(2, ec["env_cutoff_hz"], btype="low", fs=ec["fs"], output="sos").tolist()
    manifest = {
        "ear_sos": ear_sos, "env_sos": env_sos,
        "sexes": sexes, "stages": sc["stage_order"], "stage_labels": sc["stage_labels"],
        "ear": cfg["ear"], "sim": {k: v for k, v in cfg["sim"].items() if k != "w_syn_mv"},
        "stimuli": cfg["stimuli"], "config_hash": config_hash(cfg),
        "subnet_report": json.loads((DATA / "subnet_report.json").read_text()) if (DATA / "subnet_report.json").exists() else {},
        "has_sweep": res_p.exists(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), default=str))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"export: {total / 1e6:.1f} MB in {OUT.relative_to(ROOT)}; runs: " +
          ", ".join(f"{s}={len(v['runs'])}" for s, v in sexes.items()))
    if total > 150e6:
        raise SystemExit("export exceeds 150 MB")


if __name__ == "__main__":
    main()
