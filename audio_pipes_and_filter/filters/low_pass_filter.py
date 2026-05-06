# -*- coding: utf-8 -*-
"""
Einfacher Tiefpass ueber gleitenden Mittelwert.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class LowPassFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Tiefpass"},
            "window_size": {"type": int, "value": 9},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        window_size = max(1, int(self.get("window_size", 9)))
        if window_size % 2 == 0:
            window_size += 1

        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        samples = audio_data["samples"]
        filtered = np.zeros_like(samples)
        for channel in range(samples.shape[1]):
            filtered[:, channel] = np.convolve(samples[:, channel], kernel, mode="same")

        out = clone_audio(audio_data, samples=filtered, name=f"{audio_data['name']} | LowPass")
        self.last_data = out
        return out
