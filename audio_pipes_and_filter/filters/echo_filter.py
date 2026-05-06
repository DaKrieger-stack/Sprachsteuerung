# -*- coding: utf-8 -*-
"""
Fuegt dem Signal ein einfaches Echo hinzu.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class EchoFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Echo"},
            "delay_ms": {"type": float, "value": 180.0},
            "decay": {"type": float, "value": 0.45},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        samples = audio_data["samples"]
        delay_ms = max(1.0, float(self.get("delay_ms", 180.0)))
        decay = float(self.get("decay", 0.45))
        delay_samples = max(1, int(audio_data["sample_rate"] * delay_ms / 1000.0))

        out_samples = np.array(samples, copy=True)
        if delay_samples < len(out_samples):
            out_samples[delay_samples:] += samples[:-delay_samples] * decay
        out_samples = np.clip(out_samples, -1.0, 1.0)

        out = clone_audio(audio_data, samples=out_samples, name=f"{audio_data['name']} | Echo")
        self.last_data = out
        return out
