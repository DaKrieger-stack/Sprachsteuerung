# -*- coding: utf-8 -*-
"""
Einfacher Reverb-Effekt ueber mehrere abgeklungene, kurze Delays.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class ReverbFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Reverb"},
            "mix": {"type": float, "value": 0.35},
            "room_size": {"type": float, "value": 0.6},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        mix = min(1.0, max(0.0, float(self.get("mix", 0.35))))
        room_size = min(1.0, max(0.1, float(self.get("room_size", 0.6))))
        base_delays = [0.011, 0.019, 0.031, 0.047]
        samples = audio_data["samples"]
        wet = np.zeros_like(samples)

        for idx, delay_seconds in enumerate(base_delays):
            delay = max(1, int(audio_data["sample_rate"] * delay_seconds * room_size))
            decay = 0.55 / (idx + 1)
            if delay < len(samples):
                wet[delay:] += samples[:-delay] * decay

        out_samples = np.clip((1.0 - mix) * samples + mix * wet, -1.0, 1.0)
        out = clone_audio(audio_data, samples=out_samples, name=f"{audio_data['name']} | Reverb")
        self.last_data = out
        return out
