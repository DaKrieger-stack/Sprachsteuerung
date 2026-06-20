#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import tensorflow as tf

from export_esp32_model import CLASSES, load_xy

TFLITE = Path(__file__).resolve().parents[1] / "models" / "esp32_voice_int8.tflite"


def main():
    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print("input", inp["shape"], inp["dtype"], inp.get("quantization"))
    print("output", out["shape"], out["dtype"], out.get("quantization"))

    x, y = load_xy()
    for label in CLASSES:
        for idx in np.where(y == label)[0][:3]:
            interp.set_tensor(inp["index"], x[idx : idx + 1])
            interp.invoke()
            probs = interp.get_tensor(out["index"])[0]
            print(f"{CLASSES[label]:9s} true={label} probs={probs}")


if __name__ == "__main__":
    main()
