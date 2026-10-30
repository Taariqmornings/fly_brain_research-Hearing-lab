"""Dataset access: flyconn stores + neuron metadata with 3D positions (um)."""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_config() -> dict:
    return yaml.safe_load((ROOT / "pipeline" / "config.yaml").read_text())


def _env(cfg: dict) -> None:
    os.environ.setdefault("FLYCONN_CACHE", str(ROOT / cfg["data"]["cache"]))


def ensure(cfg: dict) -> None:
    """Pull both datasets at weights level (no-op when cached) and FlyWire positions."""
    _env(cfg)
    from flyconn.data.cache import cache_root  # noqa: F401  (imported after env is set)

    exe = Path(sys.executable).with_name("flyconn.exe" if os.name == "nt" else "flyconn")
    for ref in cfg["datasets"].values():
        store = ROOT / cfg["data"]["cache"] / "store" / ref.replace("@", "/")
        if not (store / "provenance.json").exists():
            subprocess.run([str(exe), "data", "pull", ref, "--level", "weights"], check=True)
    fw = DATA / "positions" / "fw_annotations.tsv"
    if not fw.exists():
        fw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(cfg["data"]["fw_positions_url"], fw)


def store(ref: str, cfg: dict):
    _env(cfg)
    from flyconn.data.store import Store

    return Store.open(ref)


def _xyz(v) -> tuple[float, float, float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return (np.nan, np.nan, np.nan)
    a = np.asarray(v, dtype=float).ravel()
    return tuple(a[:3]) if a.size >= 3 else (np.nan, np.nan, np.nan)


def neurons(ref: str, cfg: dict) -> pd.DataFrame:
    """neuron_id, cell_type, label (type + synonyms), nt_pred, nt_conf, super_class, side, x, y, z (um)."""
    cache = DATA / f"neurons_{ref.replace('@', '_')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    s = store(ref, cfg)
    df = s.neurons(["neuron_id", "cell_type", "nt_pred", "nt_conf", "super_class", "side"])
    if ref.startswith("flywire"):
        a = pd.read_csv(DATA / "positions" / "fw_annotations.tsv", sep="\t", low_memory=False,
                        usecols=["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "synonyms"])
        vx = np.array(cfg["data"]["fw_voxel_nm"], dtype=float) / 1000.0
        has = a.soma_x.notna()
        xyz = np.where(has.values[:, None], a[["soma_x", "soma_y", "soma_z"]].values,
                       a[["pos_x", "pos_y", "pos_z"]].values).astype(float) * vx
        pos = pd.DataFrame({"neuron_id": a.root_id.astype("int64"), "x": xyz[:, 0], "y": xyz[:, 1],
                            "z": xyz[:, 2], "syn": a.synonyms})
    else:
        import pyarrow.feather as feather

        raw = next((ROOT / cfg["data"]["cache"] / "raw" / "malecns" / "1.0").glob("body-annotations*.feather"))
        a = feather.read_table(raw, columns=["bodyId", "somaLocation", "tosomaLocation", "synonyms",
                                              "instance"]).to_pandas()
        loc = [(_xyz(s) if s is not None else _xyz(t)) for s, t in zip(a.somaLocation, a.tosomaLocation)]
        xyz = np.array(loc, dtype=float) * 0.008
        syn = a.synonyms.fillna("") + " " + a.instance.fillna("")
        pos = pd.DataFrame({"neuron_id": a.bodyId.astype("int64"), "x": xyz[:, 0], "y": xyz[:, 1],
                            "z": xyz[:, 2], "syn": syn})
    df = df.merge(pos.drop_duplicates("neuron_id"), on="neuron_id", how="left")
    df["cell_type"] = df.cell_type.fillna("")
    df["label"] = (df.cell_type + " | " + df.syn.fillna("")).str.strip()
    df = df.drop(columns="syn")
    df.to_parquet(cache)
    return df


def counts(ref: str, cfg: dict) -> tuple[int, int, int]:
    s = store(ref, cfg)
    n = int(s.query("select count(*) n from neurons").n[0])
    e = s.query("select count(*) e, sum(weight) w from edges")
    return n, int(e.e[0]), int(e.w[0])


def version_info(ref: str, cfg: dict) -> dict:
    import hashlib
    import json

    p = store(ref, cfg).provenance
    return {"ref": ref, "provenance_sha": hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:12]}


if __name__ == "__main__":
    cfg = load_config()
    ensure(cfg)
    for sex, ref in cfg["datasets"].items():
        n, e, w = counts(ref, cfg)
        nd = neurons(ref, cfg)
        print(f"{sex} {ref}: neurons={n} edges={e} synapses={w} with_position={int(nd.x.notna().sum())}")
