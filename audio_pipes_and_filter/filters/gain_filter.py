# -*- coding: utf-8 -*-
"""
Verstaerkt oder schwaecht ein Audiosignal.
"""

import numpy as np

from filters.base_filter import BaseFilter
from utils.audio_utils import clone_audio


class VolumeFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Lautstaerke"},
            "gain_db": {"type": float, "value": 6.0},
        }

    def process(self, audio_data):
        if audio_data is None:
            self.last_data = None
            return None
        gain_db = self.get("gain_db", 0.0)
        factor = float(10.0 ** (gain_db / 20.0))
        out = clone_audio(
            audio_data,
            samples=np.clip(audio_data["samples"] * factor, -1.0, 1.0),
            name=f"{audio_data['name']} | Volume",
        )
        self.last_data = out
        return out


class GainFilter(VolumeFilter):
    pass
