# -*- coding: utf-8 -*-
"""
Einfacher Frequenzfilter mit Low-pass- oder High-pass-Modus.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class FrequencyFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Frequenzfilter"},
            "mode": {"type": str, "value": "low"},
            "window_size": {"type": int, "value": 11},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        window_size = max(1, int(self.get("window_size", 11)))
        if window_size % 2 == 0:
            window_size += 1
        mode = str(self.get("mode", "low")).strip().lower()

        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        samples = audio_data["samples"]
        low_pass = np.zeros_like(samples)
        for channel in range(samples.shape[1]):
            low_pass[:, channel] = np.convolve(samples[:, channel], kernel, mode="same")

        if mode == "high":
            filtered = samples - low_pass
            suffix = "HighPass"
        else:
            filtered = low_pass
            suffix = "LowPass"

        out = clone_audio(
            audio_data,
            samples=np.clip(filtered, -1.0, 1.0),
            name=f"{audio_data['name']} | {suffix}",
        )
        self.last_data = out
        return out
