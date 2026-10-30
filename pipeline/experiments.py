"""IPI x carrier sweeps with automatic controls, robustness statistics and validation check."""
from __future__ import annotations

import argparse
import shutil
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import DATA, load_config, version_info
from ear import audio_hash, stimulus
from sim import build_net, config_hash, git_commit, load_subnet, run_meta, simulate

RESULTS = DATA / "results"
VARIANTS = ["baseline", "shuffled", "thr10", "signflip"]
REFS = {"pulse35": ("pulse", {}), "scrambled35": ("scrambled", {}), "noise": ("noise", {}), "sine": ("sine", {})}


def make_net(sex: str, variant: str, cfg: dict, sub: dict):
    if variant == "shuffled":
        return build_net(sex, cfg, sub, shuffle_seed=12345)
    if variant == "thr10":
        return build_net(sex, cfg, sub, min_weight=cfg["experiments"]["weight_threshold_alt"])
    if variant == "signflip":
        return build_net(sex, cfg, sub, flip_low_conf=True)
    return build_net(sex, cfg, sub)


def conditions(grid: dict, variant: str) -> list[dict]:
    out = [{"id": f"pulse_ipi{i}_f{f}", "kind": "pulse", "ipi_ms": i, "carrier_hz": f, "grid": "pulse"}
           for i in grid["ipi_ms"] for f in grid["carrier_hz"]]
    if variant == "baseline":  # phase-scrambled audio control over the whole grid
        out += [{"id": f"scr_ipi{i}_f{f}", "kind": "scrambled", "ipi_ms": i, "carrier_hz": f, "grid": "scrambled"}
                for i in grid["ipi_ms"] for f in grid["carrier_hz"]]
    out += [{"id": k, "kind": kind, "grid": "ref", **kw} for k, (kind, kw) in REFS.items()]
    return out


def run_variant(sex: str, variant: str, cfg: dict, sub: dict, grid: dict, seeds: list[int], batch: int) -> dict:
    net = make_net(sex, variant, cfg, sub)
    conds = conditions(grid, variant)
    trials = [(c, s) for c in conds for s in seeds]
    rates = {c["id"]: {} for c in conds}
    stage = {c["id"]: {} for c in conds}
    runs_dir = DATA / "runs" / sex
    runs_dir.mkdir(parents=True, exist_ok=True)
    rec = variant == "baseline"
    for b0 in range(0, len(trials), batch):
        chunk = trials[b0:b0 + batch]
        audios = [stimulus(c["kind"], cfg, seed=s, duration_s=grid["duration_s"],
                           **{k: c[k] for k in ("ipi_ms", "carrier_hz") if k in c}) for c, s in chunk]
        r = simulate(net, audios, [s for _, s in chunk], cfg, record=rec)
        for k, (c, s) in enumerate(chunk):
            rates[c["id"]][s] = float(r["rates"][k][net.target].mean())
            stage[c["id"]][s] = r["stage_rates"][k].mean(axis=1).tolist()
            if rec and s == seeds[0]:
                meta = run_meta(net, cfg, audios[k], s, condition=c, duration_s=grid["duration_s"])
                np.savez_compressed(runs_dir / f"{c['id']}.npz", ids=r["spikes"][k][0], t=r["spikes"][k][1],
                                    rates=r["rates"][k].astype(np.float32), stage_rates=r["stage_rates"][k].astype(np.float32),
                                    audio=audios[k].astype(np.float32), meta=json.dumps(meta))
        print(f"    {sex}/{variant}: {min(b0 + batch, len(trials))}/{len(trials)} trials", flush=True)
    return {"rates": rates, "stage_rates": stage, "variant": net.variant, "n_target": int(net.target.sum())}


def boot_ci(x: np.ndarray, n: int, rng) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return (float(x.mean()), float(x.mean()))
    m = rng.choice(x, (n, len(x))).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def effect(diff: np.ndarray, n: int, rng) -> dict:
    d = np.asarray(diff, float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    lo, hi = boot_ci(d, n, rng)
    return {"mean": float(d.mean()), "ci": [lo, hi], "d": float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf")),
            "sig": bool(lo > 0 or hi < 0), "n": int(len(d))}


def grid_array(res: dict, grid: dict, prefix: str, seeds: list[int]) -> np.ndarray:
    """(seeds, ipi, carrier) target rates."""
    return np.array([[[res["rates"][f"{prefix}_ipi{i}_f{f}"][s] for f in grid["carrier_hz"]]
                      for i in grid["ipi_ms"]] for s in seeds])


def ipi_tuning(g: np.ndarray) -> np.ndarray:
    """Per seed: max over IPI of carrier-averaged response minus its mean over IPI."""
    m = g.mean(axis=2)
    return m.max(axis=1) - m.mean(axis=1)


def analyse(all_res: dict, grid: dict, seeds: list[int], cfg: dict) -> dict:
    rng = np.random.default_rng(0)
    nb = cfg["experiments"]["bootstrap"]
    out = {"per_sex": {}, "robust": {}, "validation": {}}
    effects: dict[str, dict[str, dict]] = {}
    for sex, vr in all_res.items():
        r = {v: vr[v]["rates"] for v in vr}
        sel = lambda v, a, b: np.array([r[v][a][s] - r[v][b][s] for s in seeds])  # noqa: E731
        base = grid_array(vr["baseline"], grid, "pulse", seeds)
        scr = grid_array(vr["baseline"], grid, "scr", seeds)
        mean = base.mean(0)
        pi, pf = np.unravel_index(np.argmax(mean), mean.shape)
        flat = bool(np.allclose(mean, mean.flat[0]))
        out["per_sex"][sex] = {
            "grid_mean": mean.tolist(), "grid_scrambled_mean": scr.mean(0).tolist(),
            "peak": None if flat else {"ipi_ms": grid["ipi_ms"][pi], "carrier_hz": grid["carrier_hz"][pf], "rate_hz": float(mean[pi, pf])},
            "refs": {k: {"mean": float(np.mean([r["baseline"][k][s] for s in seeds])),
                         "ci": boot_ci([r["baseline"][k][s] for s in seeds], nb, rng)} for k in REFS},
            "stage_rates_pulse35": np.mean([vr["baseline"]["stage_rates"]["pulse35"][s] for s in seeds], axis=0).tolist(),
            "n_target": vr["baseline"]["n_target"],
        }
        e = {}
        for v in VARIANTS:
            e.setdefault("song_vs_scrambled", {})[v] = effect(sel(v, "pulse35", "scrambled35"), nb, rng)
            e.setdefault("song_vs_noise", {})[v] = effect(sel(v, "pulse35", "noise"), nb, rng)
            e.setdefault("ipi_tuning", {})[v] = effect(ipi_tuning(grid_array(vr[v], grid, "pulse", seeds)), nb, rng)
        e["ipi_tuning"]["scrambled_audio"] = effect(ipi_tuning(scr), nb, rng)
        effects[sex] = e
        v1, v2 = e["song_vs_scrambled"]["baseline"], e["song_vs_noise"]["baseline"]
        out["validation"][sex] = {"pulse_gt_scrambled": v1, "pulse_gt_noise": v2,
                                  "result": "PASS" if (v1["sig"] and v1["mean"] > 0 and v2["sig"] and v2["mean"] > 0) else "FAIL"}
    # Sex difference (male - female) on pulse song: independent samples, compare per seed.
    if {"male", "female"} <= set(all_res):
        effects["sex_difference"] = {}
        for v in VARIANTS:
            d = np.array([all_res["male"][v]["rates"]["pulse35"][s] - all_res["female"][v]["rates"]["pulse35"][s] for s in seeds])
            effects["sex_difference"].setdefault("pulse35_male_minus_female", {})[v] = effect(d, nb, rng)
    for group, eff in effects.items():
        for name, ev in eff.items():
            b = ev["baseline"]
            same = lambda x: x["sig"] and np.sign(x["mean"]) == np.sign(b["mean"])  # noqa: E731
            checks = {
                "baseline_significant": b["sig"],
                "survives_threshold_10": same(ev["thr10"]),
                "survives_sign_flip": same(ev["signflip"]),
                "absent_or_smaller_in_shuffled": bool(not same(ev["shuffled"]) or
                                                      (ev["shuffled"]["ci"][1] < b["ci"][0] if b["mean"] > 0 else ev["shuffled"]["ci"][0] > b["ci"][1])),
            }
            if "scrambled_audio" in ev:
                checks["absent_or_smaller_with_scrambled_audio"] = bool(ev["scrambled_audio"]["mean"] < b["mean"])
            out["robust"][f"{group}:{name}"] = {"effects": ev, "checks": checks, "robust": bool(all(checks.values()))}
    return out


def heatmaps(res: dict, grid: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sexes = list(res["per_sex"])
    fig, axes = plt.subplots(2, len(sexes), figsize=(5 * len(sexes), 8), squeeze=False)
    for j, sex in enumerate(sexes):
        for i, (key, title) in enumerate((("grid_mean", "pulse song"), ("grid_scrambled_mean", "phase-scrambled"))):
            ax = axes[i, j]
            im = ax.imshow(np.array(res["per_sex"][sex][key]), origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks(range(len(grid["carrier_hz"])), grid["carrier_hz"])
            ax.set_yticks(range(len(grid["ipi_ms"])), grid["ipi_ms"])
            ax.set_xlabel("carrier (Hz)")
            ax.set_ylabel("IPI (ms)")
            ax.set_title(f"{sex}: target rate (Hz), {title}")
            fig.colorbar(im, ax=ax)
    fig.suptitle("Model predictions - song/receptivity target set")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2x2 grid, 2 seeds, 0.2 s (debug)")
    a = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(4)
    ex = cfg["experiments"]
    mode = "smoke" if a.smoke else "quick" if a.quick else "full"
    grid = dict(ex["quick" if mode != "full" else "full"])
    seeds = list(ex["seeds"])
    if a.smoke:
        grid = {"ipi_ms": [15, 35], "carrier_hz": [150, 220], "duration_s": 0.2}
        seeds = seeds[:2]
    RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(DATA / "runs", ignore_errors=True)  # runs from a previous grid would mix into the export
    t0 = time.time()
    all_res = {}
    for sex in ("male", "female"):
        sub = load_subnet(sex)
        all_res[sex] = {v: run_variant(sex, v, cfg, sub, grid, seeds, ex["cpu_trial_batch"]) for v in VARIANTS}
    res = analyse(all_res, grid, seeds, cfg)
    res["meta"] = {"mode": mode, "grid": grid, "seeds": seeds, "config_hash": config_hash(cfg), "git_commit": git_commit(),
                   "datasets": {s: version_info(r, cfg) for s, r in cfg["datasets"].items()},
                   "runtime_s": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "ref_audio_hash": audio_hash(stimulus("pulse", cfg, duration_s=grid["duration_s"]))}
    res["raw"] = {sex: {v: all_res[sex][v]["rates"] for v in VARIANTS} for sex in all_res}
    (RESULTS / "results.json").write_text(json.dumps(res, indent=1, default=float))
    heatmaps(res, grid, RESULTS / "heatmaps.png")
    for sex, p in res["per_sex"].items():
        pk = p["peak"]
        print(f"{sex}: peak " + (f"IPI {pk['ipi_ms']} ms / carrier {pk['carrier_hz']} Hz ({pk['rate_hz']:.2f} Hz)" if pk else "none (flat: no target response)")
              + f"; validation {res['validation'][sex]['result']}")
    for k, v in res["robust"].items():
        print(f"  {k}: robust={v['robust']} baseline={v['effects']['baseline']['mean']:.3f} {v['effects']['baseline']['ci']}")
    print(f"runtime {res['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
