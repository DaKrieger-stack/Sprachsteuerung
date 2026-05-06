# -*- coding: utf-8 -*-
"""
Veraendert die Tonhoehe naeherungsweise ueber Resampling und Rueckskalierung.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class PitchShiftFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Pitch"},
            "semitones": {"type": float, "value": 3.0},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None

        semitones = float(self.get("semitones", 3.0))
        factor = float(2.0 ** (semitones / 12.0))
        samples = audio_data["samples"]
        old_len = len(samples)
        compressed_len = max(1, int(old_len / factor))

        src_positions = np.arange(old_len, dtype=np.float32)
        mid_positions = np.linspace(0, old_len - 1, compressed_len, dtype=np.float32)
        resampled = np.zeros((compressed_len, samples.shape[1]), dtype=np.float32)
        for channel in range(samples.shape[1]):
            resampled[:, channel] = np.interp(mid_positions, src_positions, samples[:, channel])

        back_positions = np.arange(compressed_len, dtype=np.float32)
        target_positions = np.linspace(0, compressed_len - 1, old_len, dtype=np.float32)
        shifted = np.zeros_like(samples)
        for channel in range(samples.shape[1]):
            shifted[:, channel] = np.interp(target_positions, back_positions, resampled[:, channel])

        out = clone_audio(
            audio_data,
            samples=np.clip(shifted, -1.0, 1.0),
            name=f"{audio_data['name']} | Pitch",
        )
        self.last_data = out
        return out
