#!/usr/bin/env python3
"""Train and export a small 3-class int8 model that fits ESP32 TFLite arena (~120 KB)."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUT_TFLITE = ROOT / "ESP_Code" / "ESP_Code" / "models" / "esp32_voice_int8.tflite"
OUT_MODELL_C = ROOT / "ESP_Code" / "ESP_Code" / "src" / "modell.c"

SR = 16000
DURATION = 2.0
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
REFERENCE_WIDTH = 124
NUM_CLASSES = 3
CLASS_DIRS = {0: "licht_an", 1: "licht_aus", 2: "unknown"}
TRIM_TOP_DB = 25
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def collect_wav_files(class_dir: Path) -> list[Path]:
    files = sorted(class_dir.glob("*.wav"))
    return files if files else sorted(class_dir.rglob("*.wav"))


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).flatten())
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / peak
    return audio


def prepare_audio_clip(audio: np.ndarray, trim_silence: bool = True) -> np.ndarray:
    audio = normalize_audio(audio)
    if trim_silence:
        trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
        if len(trimmed) > 0:
            audio = trimmed

    target_len = int(SR * DURATION)
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start : start + target_len]
    elif len(audio) < target_len:
        pad_total = target_len - len(audio)
        audio = np.pad(audio, (pad_total // 2, pad_total - pad_total // 2))

    return audio.astype(np.float32)


def get_mel_spec(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def pad_or_crop_time(spec: np.ndarray, target_width: int = REFERENCE_WIDTH) -> np.ndarray:
    width = spec.shape[1]
    if width > target_width:
        start = (width - target_width) // 2
        spec = spec[:, start : start + target_width]
    elif width < target_width:
        pad = target_width - width
        spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant")
    return spec


def file_to_input(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    prepared = prepare_audio_clip(audio, trim_silence=True)
    spec = pad_or_crop_time(get_mel_spec(prepared))
    return spec[..., np.newaxis]


def build_esp32_cnn(input_shape=(N_MELS, REFERENCE_WIDTH, 1)):
    # Small filter counts so largest int8 activation stays below ~65 KB.
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(24, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def load_dataset():
    xs, ys = [], []
    for label_idx, folder in CLASS_DIRS.items():
        for path in collect_wav_files(DATA_DIR / folder):
            xs.append(file_to_input(path))
            ys.append(label_idx)
    if not xs:
        raise FileNotFoundError(f"No wav files under {DATA_DIR}")
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def representative_dataset(samples: np.ndarray):
    for i in range(min(len(samples), 100)):
        yield [samples[i : i + 1]]


def write_modell_c(data: bytes, dst: Path):
    lines = ["const unsigned char tiny_cnn_model_big_tflite[] = {"]
    row = []
    for i, b in enumerate(data):
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


def main():
    print("Loading dataset...")
    x, y = load_dataset()
    print(f"Samples: {len(x)}, shape: {x.shape[1:]}")

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = build_esp32_cnn(tuple(x.shape[1:]))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weight = {i: float(len(y_train)) / (NUM_CLASSES * max(c, 1)) for i, c in enumerate(counts)}

    print("Training ESP32 model...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=40,
        batch_size=8,
        class_weight=class_weight,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.3f}")

    print("Exporting int8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    OUT_TFLITE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TFLITE.write_bytes(tflite_model)
    write_modell_c(tflite_model, OUT_MODELL_C)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    max_bytes = 0
    max_name = ""
    for t in interpreter.get_tensor_details():
        sh = t["shape"]
        if sh is None or len(sh) == 0:
            continue
        n = int(np.prod(sh))
        b = n * (4 if t["dtype"] == np.float32 else 1)
        if b > max_bytes:
            max_bytes = b
            max_name = t["name"]
    print(f"Saved: {OUT_TFLITE} ({len(tflite_model)} bytes)")
    print(f"Updated: {OUT_MODELL_C}")
    print(f"Input: {inp['shape']} {inp['dtype']} | Output: {out['shape']} {out['dtype']}")
    print(f"Largest tensor ~{max_bytes} bytes ({max_name})")

    mapping = {str(i): CLASS_DIRS[i] for i in range(NUM_CLASSES)}
    (ROOT / "label_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
