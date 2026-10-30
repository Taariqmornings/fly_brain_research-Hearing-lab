"""Hearing subnetwork: JO seeds + strongest downstream paths within N hops, plus target cell types."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from data import DATA, load_config, neurons, store
from ear import subtype_of


def _match(labels: pd.Series, patterns: list[str]) -> np.ndarray:
    rx = re.compile("|".join(f"(?:{p})" for p in patterns), re.I)
    return labels.fillna("").map(lambda s: bool(rx.search(s))).values


def _sign(nt: str | None, table: dict) -> int:
    return int(table.get(nt, table["default"])) if nt else int(table["default"])


def build(sex: str, cfg: dict) -> dict:
    ref = cfg["datasets"][sex]
    sc = cfg["subnet"]
    s = store(ref, cfg)
    nd = neurons(ref, cfg).set_index("neuron_id")
    seeds = nd.index[nd.cell_type.str.match(sc["seed_regex"])].values
    in_tot = s.query("select post, sum(weight) w from edges group by post").set_index("post").w
    score = pd.Series(1.0, index=seeds)
    hop = pd.Series(0, index=seeds)
    frontier = seeds
    for h in range(1, sc["hops"] + 1):
        e = s.edges(min_weight=sc["min_weight"], pre=frontier.tolist())
        e = e[["pre", "post", "weight"]]
        e["s"] = score.reindex(e.pre).values * e.weight.values / in_tot.reindex(e.post).values
        best = e.groupby("post").s.max()
        new = best.index.difference(hop.index)
        score = pd.concat([score, best]).groupby(level=0).max()
        hop = pd.concat([hop, pd.Series(h, index=new)])
        frontier = new.values
        print(f"  {sex} hop {h}: +{len(new)} neurons")
    labels = nd.label
    tmask = {k: _match(labels, v) for k, v in sc["targets"].items()}
    target_ids = {k: nd.index[m].values for k, m in tmask.items()}
    all_t = np.unique(np.concatenate(list(target_ids.values()))) if sc["include_targets"] else np.array([], int)
    # Strongest JO->target paths first: backward input-fraction score from the targets, times forward score.
    back = pd.Series(1.0, index=all_t)
    frontier = all_t
    for _ in range(sc["hops"]):
        if not len(frontier):
            break
        e = s.edges(min_weight=sc["min_weight"], post=frontier.tolist())[["pre", "post", "weight"]]
        e["s"] = back.reindex(e.post).values * e.weight.values / in_tot.reindex(e.post).values
        best = e.groupby("pre").s.max()
        frontier = best.index.difference(back.index).values
        back = pd.concat([back, best]).groupby(level=0).max()
    excl = np.concatenate([seeds, all_t])
    path = (score * back.reindex(score.index).fillna(0)).drop(index=excl, errors="ignore")
    path = path[path > 0].sort_values(ascending=False)
    room = max(0, sc["max_neurons"] - len(seeds) - len(all_t))
    on_path = path.index.values[: room // 2]
    rest = score.drop(index=np.concatenate([excl, on_path]), errors="ignore").sort_values(ascending=False)
    keep = np.unique(np.concatenate([seeds, all_t, on_path, rest.index.values[: room - len(on_path)]])).astype(np.int64)
    print(f"  {sex}: {len(on_path)} neurons kept for JO->target paths")
    keep = keep[np.isin(keep, nd.index.values)]
    idx = pd.Series(np.arange(len(keep)), index=keep)
    e = s.edges(min_weight=sc["min_weight"], pre=keep.tolist())
    e = e[e.post.isin(keep)][["pre", "post", "weight"]]
    e = e.groupby(["pre", "post"], as_index=False).weight.sum()
    sub = nd.loc[keep]
    hops = hop.reindex(keep).fillna(-1).astype(int).values
    stage = np.array([""] * len(keep), dtype=object)
    for name, rx in sc["stages"]:
        m = (stage == "") & sub.label.str.contains(rx, case=False, regex=True).values
        stage[m] = name
    for h, name in sc["hop_stage"].items():
        stage[(stage == "") & (hops == int(h))] = name
    stage[stage == ""] = "song"
    signs = np.array([_sign(v, cfg["signs"]) for v in sub.nt_pred.values], dtype=np.int8)
    pre_i, post_i = idx[e.pre].values, idx[e.post].values
    xyz = _fill_positions(sub[["x", "y", "z"]].values.astype(float), pre_i, post_i, stage)
    out = dict(
        ids=keep, cell_type=np.asarray(sub.cell_type.tolist(), dtype="U"), label=np.asarray(sub.label.tolist(), dtype="U"),
        stage=stage.astype(str), hop=hops, sign=signs, nt=np.asarray(sub.nt_pred.fillna("").tolist(), dtype="U"),
        nt_conf=sub.nt_conf.fillna(0).values.astype(np.float32), xyz=xyz.astype(np.float32),
        pre=pre_i.astype(np.int32), post=post_i.astype(np.int32), weight=e.weight.values.astype(np.int32),
        jo_subtype=np.array([subtype_of(c) if st == "jo" else "" for c, st in zip(sub.cell_type, stage)]),
        is_song=np.isin(keep, target_ids["song_detectors"]), is_recep=np.isin(keep, target_ids["receptivity"]),
        pos_measured=sub.x.notna().values, dataset=np.array(ref),
    )
    np.savez_compressed(DATA / f"subnet_{sex}.npz", **out)
    jo = pd.Series(out["jo_subtype"][stage == "jo"]).value_counts().sort_index().to_dict()
    report = {
        "dataset": ref, "n_neurons": int(len(keep)), "n_edges": int(len(e)), "n_seeds": int(len(seeds)),
        "jo_subtypes": {k: int(v) for k, v in jo.items()},
        "jo_types": nd.loc[seeds].cell_type.value_counts().to_dict(),
        "stage_counts": pd.Series(stage).value_counts().to_dict(),
        "targets": {k: {"in_dataset": int(len(v)), "within_hops": int(np.isin(v, hop.index).sum()),
                        "types": sorted(set(nd.loc[v].cell_type))[:40]} for k, v in target_ids.items()},
        "positions_measured": int(out["pos_measured"].sum()),
    }
    return report


def _fill_positions(xyz: np.ndarray, pre: np.ndarray, post: np.ndarray, stage: np.ndarray) -> np.ndarray:
    """Fallback layout for neurons without a soma/position: mean of positioned partners, then stage centroid."""
    rng = np.random.default_rng(0)
    xyz = xyz.copy()
    for _ in range(4):
        miss = np.isnan(xyz[:, 0])
        if not miss.any():
            break
        acc = np.zeros_like(xyz)
        cnt = np.zeros(len(xyz))
        for a, b in ((pre, post), (post, pre)):
            ok = ~np.isnan(xyz[b, 0]) & miss[a]
            np.add.at(acc, a[ok], xyz[b[ok]])
            np.add.at(cnt, a[ok], 1)
        fill = miss & (cnt > 0)
        xyz[fill] = acc[fill] / cnt[fill, None] + rng.normal(0, 3.0, (fill.sum(), 3))
    miss = np.isnan(xyz[:, 0])
    if miss.any():
        c = np.nanmean(xyz, axis=0) if (~miss).any() else np.zeros(3)
        for st in np.unique(stage[miss]):
            m = miss & (stage == st)
            base = np.nanmean(xyz[(stage == st) & ~miss], axis=0) if ((stage == st) & ~miss).any() else c
            xyz[m] = base + rng.normal(0, 10.0, (m.sum(), 3))
    return xyz


if __name__ == "__main__":
    cfg = load_config()
    reports = {}
    for sex in ("male", "female"):
        r = build(sex, cfg)
        reports[sex] = r
        print(f"{sex}: subnet={r['n_neurons']} edges={r['n_edges']} JO={r['jo_subtypes']}")
        for k, v in r["targets"].items():
            print(f"  {k}: in_dataset={v['in_dataset']} within_hops={v['within_hops']} types={v['types'][:12]}")
    (DATA / "subnet_report.json").write_text(json.dumps(reports, indent=1, default=str))
