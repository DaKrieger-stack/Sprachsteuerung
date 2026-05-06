# -*- coding: utf-8 -*-
"""
Normiert das Signal auf einen Ziel-Peakwert.
"""

from filters.base_filter import BaseFilter
from utils.audio_utils import audio_peak, clone_audio


class NormalizeFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Normalisieren"},
            "target_peak": {"type": float, "value": 0.92},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        current_peak = audio_peak(audio_data)
        if current_peak <= 1e-9:
            out = clone_audio(audio_data, name=f"{audio_data['name']} | Normalize")
        else:
            factor = float(self.get("target_peak", 0.92)) / current_peak
            out = clone_audio(
                audio_data,
                samples=audio_data["samples"] * factor,
                name=f"{audio_data['name']} | Normalize",
            )
        self.last_data = out
        return out
