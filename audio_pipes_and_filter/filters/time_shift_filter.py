# -*- coding: utf-8 -*-
"""
Verschiebt das Audiosignal zeitlich nach vorne oder hinten.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class TimeShiftFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Zeitverschiebung"},
            "shift_ms": {"type": float, "value": 250.0},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        shift_ms = float(self.get("shift_ms", 250.0))
        shift_samples = int(audio_data["sample_rate"] * abs(shift_ms) / 1000.0)
        samples = audio_data["samples"]
        shifted = np.zeros_like(samples)

        if shift_samples == 0:
            shifted = np.array(samples, copy=True)
        elif shift_ms >= 0:
            if shift_samples < len(samples):
                shifted[shift_samples:] = samples[:-shift_samples]
        else:
            if shift_samples < len(samples):
                shifted[:-shift_samples] = samples[shift_samples:]

        out = clone_audio(audio_data, samples=shifted, name=f"{audio_data['name']} | Shift")
        self.last_data = out
        return out
