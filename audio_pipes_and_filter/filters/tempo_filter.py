# -*- coding: utf-8 -*-
"""
Aendert die Wiedergabegeschwindigkeit bzw. das Tempo ueber Resampling.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class TempoFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Tempo"},
            "speed_factor": {"type": float, "value": 1.2},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        speed_factor = max(0.1, float(self.get("speed_factor", 1.2)))
        samples = audio_data["samples"]
        old_len = len(samples)
        new_len = max(1, int(old_len / speed_factor))
        source_positions = np.arange(old_len, dtype=np.float32)
        target_positions = np.linspace(0, old_len - 1, new_len, dtype=np.float32)

        out_samples = np.zeros((new_len, samples.shape[1]), dtype=np.float32)
        for channel in range(samples.shape[1]):
            out_samples[:, channel] = np.interp(target_positions, source_positions, samples[:, channel])

        out = clone_audio(audio_data, samples=out_samples, name=f"{audio_data['name']} | Tempo")
        self.last_data = out
        return out
