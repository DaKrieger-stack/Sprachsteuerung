# -*- coding: utf-8 -*-
"""
Hilfsfunktionen fuer PCM-WAV-Dateien und Demo-Audiodaten.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def load_wav(path: str) -> dict:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Es werden nur PCM-WAV-Dateien mit 16 Bit unterstuetzt.")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    else:
        samples = samples.reshape(-1, 1)

    normalized = samples / 32768.0
    return {
        "name": Path(path).name,
        "path": path,
        "sample_rate": sample_rate,
        "channels": channels,
        "samples": normalized,
    }


def save_wav(audio_data: dict, path: str) -> str:
    samples = np.clip(audio_data["samples"], -1.0, 1.0)
    int_samples = (samples * 32767.0).astype(np.int16)
    flat = int_samples.reshape(-1)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(audio_data["channels"])
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio_data["sample_rate"])
        wav_file.writeframes(flat.tobytes())

    return path


def clone_audio(audio_data: dict, *, samples=None, name: str | None = None, path: str | None = None) -> dict:
    copied = dict(audio_data)
    copied["samples"] = np.array(samples if samples is not None else audio_data["samples"], dtype=np.float32, copy=True)
    if name is not None:
        copied["name"] = name
    if path is not None:
        copied["path"] = path
    return copied


def audio_duration(audio_data: dict) -> float:
    return float(len(audio_data["samples"])) / float(audio_data["sample_rate"])


def audio_peak(audio_data: dict) -> float:
    samples = audio_data["samples"]
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def audio_rms(audio_data: dict) -> float:
    samples = audio_data["samples"]
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def mono_mix(audio_data: dict) -> np.ndarray:
    samples = audio_data["samples"]
    if samples.ndim == 1 or samples.shape[1] == 1:
        return samples.reshape(-1)
    return np.mean(samples, axis=1)


def create_demo_audio(path: str, duration_seconds: float = 3.5, sample_rate: int = 44100) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = int(duration_seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / float(sample_rate)

    melody = (
        0.42 * np.sin(2.0 * math.pi * 220.0 * t) +
        0.28 * np.sin(2.0 * math.pi * 330.0 * t) +
        0.18 * np.sin(2.0 * math.pi * 440.0 * t)
    )
    envelope = np.minimum(1.0, t / 0.2) * np.minimum(1.0, (duration_seconds - t) / 0.5)
    click = np.where((t % 0.5) < 0.015, 0.22, 0.0)
    noise = 0.015 * np.random.default_rng(7).standard_normal(total)

    mono = np.clip((melody * envelope) + click + noise, -1.0, 1.0).astype(np.float32)
    stereo = np.column_stack((mono, mono * 0.93))
    audio_data = {
        "name": out_path.name,
        "path": str(out_path),
        "sample_rate": sample_rate,
        "channels": 2,
        "samples": stereo,
    }
    save_wav(audio_data, str(out_path))
    return str(out_path)
