#!/usr/bin/env python3
"""Train + export int8 TFLite for ESP32. Preprocessing mirrors src/main.cpp exactly."""

from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OUT_TFLITE = ROOT / "ESP_Code" / "ESP_Code" / "models" / "esp32_voice_int8.tflite"
OUT_C = ROOT / "ESP_Code" / "ESP_Code" / "src" / "modell.c"

SR, DUR = 16000, 2.0
N_MELS, N_FFT, HOP, WIDTH = 64, 512, 256, 124
SAMPLES = int(SR * DUR)
CLASSES = {0: "licht_an", 1: "licht_aus"}
CLASS_SOURCES = {
    0: ["licht_an", "licht_an_basic"],
    1: ["licht_aus", "licht_aus_basic"],
}
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --- ESP32-matching audio front-end (keep in sync with main.cpp) ---


def prep_audio(audio: np.ndarray) -> np.ndarray:
    pre = 0.97
    out = audio.astype(np.float32).copy()
    for i in range(len(out) - 1, 0, -1):
        v = out[i] - pre * out[i - 1]
        out[i] = np.clip(v, -1.0, 1.0)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= 32767.0 / peak
        out /= 32768.0
    return out


def trim_silence(audio: np.ndarray, thresh_db: float = -40.0) -> np.ndarray:
    t = 10.0 ** (thresh_db / 20.0)
    start, end = 0, len(audio)
    for i, s in enumerate(audio):
        if abs(s) > t:
            start = i
            break
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > t:
            end = i + 1
            break
    if end <= start:
        return audio
    return audio[start:end]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


_MEL_FB: np.ndarray | None = None


def mel_filterbank() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is not None:
        return _MEL_FB
    lo = hz_to_mel(0.0)
    hi = hz_to_mel(SR / 2.0)
    step = (hi - lo) / (N_MELS + 1)
    n_bins = N_FFT // 2 + 1
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(N_MELS):
        for k in range(n_bins):
            f = k * SR / N_FFT
            left = mel_to_hz(lo + m * step)
            mid = mel_to_hz(lo + (m + 1) * step)
            right = mel_to_hz(lo + (m + 2) * step)
            if left <= f <= mid and mid > left:
                fb[m, k] = (f - left) / (mid - left)
            elif mid <= f <= right and right > mid:
                fb[m, k] = (right - f) / (right - mid)
    _MEL_FB = fb
    return fb


def mel_spec(audio: np.ndarray, max_frames: int = WIDTH) -> np.ndarray:
    hann = np.hanning(N_FFT).astype(np.float32)
    frames = max(1, min(max_frames, (len(audio) - N_FFT) // HOP + 1))
    cols = []
    for fr in range(frames):
        start = fr * HOP
        window = np.zeros(N_FFT, dtype=np.float32)
        n = min(N_FFT, max(0, len(audio) - start))
        if n:
            window[:n] = audio[start : start + n] * hann[:n]
        spectrum = np.fft.rfft(window, n=N_FFT)
        cols.append(spectrum.real ** 2 + spectrum.imag ** 2)
    power = np.stack(cols, axis=1).astype(np.float32)  # (bins, frames)
    mel_e = mel_filterbank() @ power
    mel_e = np.maximum(mel_e, 1e-10)
    spec = 10.0 * np.log10(mel_e)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec.astype(np.float32)


def fit_spec_width(spec: np.ndarray, width: int = WIDTH) -> np.ndarray:
    cur = spec.shape[1]
    if cur == width:
        return spec
    out = np.zeros((N_MELS, width), dtype=np.float32)
    if cur > width:
        off = (cur - width) // 2
        out[:] = spec[:, off : off + width]
    else:
        out[:, :cur] = spec
    return out


def to_fixed_length(audio: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    audio = np.nan_to_num(audio.astype(np.float32).flatten())
    if len(audio) >= SAMPLES:
        # ponytail: Live-Aufnahme = Sprache oft am Anfang, nicht zentriert
        if rng is not None:
            max_start = min(len(audio) - SAMPLES, int(0.35 * len(audio)))
            start = int(rng.integers(0, max_start + 1))
        else:
            start = (len(audio) - SAMPLES) // 2
        audio = audio[start : start + SAMPLES]
    else:
        pad = SAMPLES - len(audio)
        if rng is not None:
            pad_before = int(rng.integers(0, pad + 1))
        else:
            pad_before = pad // 2
        audio = np.pad(audio, (pad_before, pad - pad_before))
    return audio


def esp32_features(audio: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    audio = prep_audio(to_fixed_length(audio, rng))
    audio = trim_silence(audio, -40.0)
    spec = fit_spec_width(mel_spec(audio))
    return spec[..., np.newaxis].astype(np.float32)


def wavs(*folders: str) -> list[Path]:
    out: list[Path] = []
    for name in folders:
        d = DATA / name
        if not d.is_dir():
            continue
        found = sorted(d.glob("*.wav")) or sorted(d.rglob("*.wav"))
        out.extend(found)
    return out


def augment(audio: np.ndarray) -> np.ndarray:
    """Light aug — gain/noise only; mel pipeline stays ESP32-identical."""
    audio = audio.copy()
    audio *= np.random.uniform(0.7, 1.3)
    if np.random.rand() < 0.5:
        audio += np.random.normal(0, np.random.uniform(0.001, 0.008), audio.shape)
    return np.clip(audio, -1.0, 1.0)


def load_xy():
    rng = np.random.default_rng(SEED)
    buckets: dict[int, list[tuple[np.ndarray, int]]] = {0: [], 1: []}
    for label, folders in CLASS_SOURCES.items():
        for path in wavs(*folders):
            raw, _ = librosa.load(path, sr=SR, mono=True)
            buckets[label].append((raw, label))
            buckets[label].append((augment(raw), label))
            if label == 1:
                buckets[label].append((augment(raw), label))
            if rng.random() < 0.5:
                buckets[label].append((augment(raw), label))

    n_cmd = min(len(buckets[0]), len(buckets[1]))
    rng.shuffle(buckets[0])
    rng.shuffle(buckets[1])
    buckets[0] = buckets[0][:n_cmd]
    buckets[1] = buckets[1][:n_cmd]

    xs, ys = [], []
    for label in CLASSES:
        for raw, lbl in buckets[label]:
            xs.append(esp32_features(raw, rng))
            ys.append(lbl)
            if rng.random() < 0.4:
                xs.append(esp32_features(raw, rng))
                ys.append(lbl)
    if not xs:
        raise FileNotFoundError(f"no wav under {DATA}")
    return np.array(xs, np.float32), np.array(ys, np.int32)


def build_cnn(shape):
    inp = keras.Input(shape=shape)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out)


def class_weights(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y, minlength=len(CLASSES))
    total = len(y)
    w = {i: total / (len(CLASSES) * max(c, 1)) for i, c in enumerate(counts)}
    w[1] *= 1.5
    return w


def write_c(data: bytes, dst: Path):
    lines = ["const unsigned char tiny_cnn_model_big_tflite[] = {"]
    row = []
    for b in data:
        row.append(f"0x{b:02x}")
        if len(row) == 12:
            lines.append("  " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("  " + ", ".join(row))
    lines.append("};")
    lines.append(f"const unsigned int tiny_cnn_model_big_tflite_len = {len(data)};")
    lines.append("")
    dst.write_text("\n".join(lines), encoding="utf-8")


def sanity_check(model, x_va, y_va):
    probs = model.predict(x_va, verbose=0)
    preds = probs.argmax(axis=1)
    acc = (preds == y_va).mean()
    for i, name in CLASSES.items():
        mask = y_va == i
        if mask.any():
            sub = (preds[mask] == i).mean()
            conf = probs[mask, i].mean()
            print(f"  {name}: recall {sub:.2f}, avg conf {conf:.2f} ({mask.sum()} samples)")
    print(f"sanity val acc {acc:.3f}")
    return acc


def main():
    x, y = load_xy()
    print(f"dataset {x.shape}, counts { {CLASSES[i]: int((y == i).sum()) for i in CLASSES} }")

    x_tr, x_va, y_tr, y_va = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    model = build_cnn(tuple(x.shape[1:]))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cw = class_weights(y_tr)
    print(f"class weights {cw}")

    cb = [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
    ]
    model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=80,
        batch_size=16,
        class_weight=cw,
        callbacks=cb,
    )
    sanity_check(model, x_va, y_va)

    def rep_dataset():
        for i in range(min(200, len(x_tr))):
            yield [x_tr[i : i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.float32
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    blob = conv.convert()

    # ponytail: sanity on exported tflite, not just keras
    test = tf.lite.Interpreter(model_content=blob)
    test.allocate_tensors()
    tin, tout = test.get_input_details()[0], test.get_output_details()[0]
    ok = True
    for label in CLASSES:
        mask = y_va == label
        if not mask.any():
            continue
        hit = 0
        for idx in np.where(mask)[0][:20]:
            test.set_tensor(tin["index"], x_va[idx : idx + 1].astype(np.float32))
            test.invoke()
            pr = test.get_tensor(tout["index"])[0]
            if pr.argmax() == label:
                hit += 1
            if label == 1 and pr[1] < 0.05:
                print(f"warn: tflite aus~0 on val idx {idx}: {pr}")
                ok = False
        print(f"  tflite {CLASSES[label]}: top1 {hit}/{min(20, mask.sum())}")
    if not ok:
        print("warn: tflite export degraded licht_aus — retrying float weights only")
        conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
        conv2.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = conv2.convert()

    OUT_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TFLITE.write_bytes(blob)
    write_c(blob, OUT_C)
    print(f"wrote {OUT_TFLITE} ({len(blob)} B) + {OUT_C}")


if __name__ == "__main__":
    main()
