"""Ear model: waveform -> per-JO-neuron drive (0..1) at 10 kHz, plus built-in stimuli."""
from __future__ import annotations

import hashlib
import re
from math import gcd

import numpy as np
from scipy import signal


def subtype_of(cell_type: str) -> str:
    m = re.match(r"^JO-([A-Z])", cell_type or "")
    return m.group(1) if m else "other"


def resample(x: np.ndarray, fs_in: int, fs: int) -> np.ndarray:
    if fs_in == fs:
        return x.astype(np.float64)
    g = gcd(int(fs_in), int(fs))
    return signal.resample_poly(x.astype(np.float64), fs // g, fs_in // g)


def normalise(x: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.mean(x**2))
    return x / r if r > 0 else x


def band_sos(band: dict, fs: int, cf_scale: float = 1.0):
    if band.get("cf", 0) <= 0:
        return signal.butter(2, band["lowpass"], btype="low", fs=fs, output="sos")
    cf = band["cf"] * cf_scale
    bw = cf / band["q"]
    lo, hi = max(5.0, cf - bw / 2), min(fs / 2 - 10, cf + bw / 2)
    return signal.butter(2, [lo, hi], btype="band", fs=fs, output="sos")


def band_envelopes(x: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Per-subtype envelope (no jitter). Input already at cfg['fs'] and unit RMS."""
    fs = cfg["fs"]
    env_sos = signal.butter(2, cfg["env_cutoff_hz"], btype="low", fs=fs, output="sos")
    out = {}
    for name, band in cfg["bands"].items():
        y = signal.sosfilt(band_sos(band, fs), x)
        y = np.maximum(y, 0.0) if band.get("cf", 0) > 0 else np.abs(y)
        out[name] = signal.sosfilt(env_sos, y)
    return out


def jo_drive(x: np.ndarray, fs_in: int, subtypes: list[str], cfg: dict, seed: int = 0) -> np.ndarray:
    """Drive in [0, 1) for each JO neuron (rows) at each 0.1 ms step (cols)."""
    fs = cfg["fs"]
    x = normalise(resample(x, fs_in, fs))
    rng = np.random.default_rng(seed)
    bands = cfg["bands"]
    env_sos = signal.butter(2, cfg["env_cutoff_hz"], btype="low", fs=fs, output="sos")
    # Per-neuron cf jitter: quantise into a few cf scales per subtype to keep filtering cheap.
    scales = 1.0 + cfg["jitter_cf_sd"] * np.array([-1.0, 0.0, 1.0])
    cache: dict[tuple[str, int], np.ndarray] = {}
    drive = np.zeros((len(subtypes), len(x)), dtype=np.float32)
    gains = np.exp(rng.normal(0.0, cfg["jitter_gain_sd"], len(subtypes)))
    pick = rng.integers(0, len(scales), len(subtypes))
    for i, st in enumerate(subtypes):
        name = st if st in bands else "other"
        key = (name, int(pick[i]))
        if key not in cache:
            band = bands[name]
            y = signal.sosfilt(band_sos(band, fs, scales[key[1]]), x)
            y = np.maximum(y, 0.0) if band.get("cf", 0) > 0 else np.abs(y)
            cache[key] = band["gain"] * signal.sosfilt(env_sos, y)
        e = np.maximum(cache[key] * gains[i], 0.0) / cfg["drive_ref_rms"]
        drive[i] = (e / (1.0 + e)).astype(np.float32)
    return drive


def ramp(x: np.ndarray, fs: int, ms: float) -> np.ndarray:
    n = min(int(fs * ms / 1000), len(x) // 2)
    if n > 0:
        w = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


def pulse_song(fs: int, dur: float, carrier: float, ipi_ms: float, cycles: float = 3.0) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    sigma = cycles / carrier / 4.0
    x = np.zeros_like(t)
    for c in np.arange(ipi_ms / 2000, dur, ipi_ms / 1000):
        x += np.exp(-0.5 * ((t - c) / sigma) ** 2) * np.sin(2 * np.pi * carrier * (t - c))
    return x


def sine_song(fs: int, dur: float, carrier: float, ramp_ms: float = 20) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    return ramp(np.sin(2 * np.pi * carrier * t), fs, ramp_ms)


def white_noise(fs: int, dur: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed + 1000).standard_normal(int(fs * dur))


def phase_scramble(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same amplitude spectrum, random phases: rhythm destroyed."""
    rng = np.random.default_rng(seed + 2000)
    f = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=len(x))


def stimulus(kind: str, cfg: dict, seed: int = 0, **kw) -> np.ndarray:
    fs, sc = cfg["ear"]["fs"], cfg["stimuli"]
    dur = kw.get("duration_s", sc["duration_s"])
    if kind == "pulse":
        return pulse_song(fs, dur, kw.get("carrier_hz", sc["pulse"]["carrier_hz"]),
                          kw.get("ipi_ms", sc["pulse"]["ipi_ms"]), sc["pulse"]["cycles"])
    if kind == "sine":
        return sine_song(fs, dur, kw.get("carrier_hz", sc["sine"]["carrier_hz"]), sc["ramp_ms"])
    if kind == "noise":
        return ramp(white_noise(fs, dur, seed), fs, sc["ramp_ms"])
    if kind == "scrambled":
        return phase_scramble(stimulus("pulse", cfg, seed, **kw), seed)
    raise ValueError(kind)


def audio_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]
