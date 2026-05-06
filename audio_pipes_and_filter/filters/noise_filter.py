# -*- coding: utf-8 -*-
"""
Fuegt Hintergrundrauschen hinzu.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class NoiseFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Rauschen"},
            "noise_level": {"type": float, "value": 0.03},
            "seed": {"type": int, "value": 7},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        rng = np.random.default_rng(int(self.get("seed", 7)))
        noise_level = max(0.0, float(self.get("noise_level", 0.03)))
        noise = rng.standard_normal(audio_data["samples"].shape).astype(np.float32) * noise_level
        out = clone_audio(
            audio_data,
            samples=np.clip(audio_data["samples"] + noise, -1.0, 1.0),
            name=f"{audio_data['name']} | Noise",
        )
        self.last_data = out
        return out
