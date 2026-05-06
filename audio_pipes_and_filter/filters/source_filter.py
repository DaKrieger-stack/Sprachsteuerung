# -*- coding: utf-8 -*-
"""
Quelle-Filter: Laedt eine WAV-Datei von der Festplatte.
Optional kann eine Demo-Datei erzeugt werden.
"""

from pathlib import Path
from tkinter import filedialog

from filters.base_filter import BaseFilter
from utils.audio_utils import create_demo_audio, load_wav


class SourceFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Quelle"},
            "pfad": {"type": str, "value": ""},
        }
        self.functions = {
            "Choose...": self.choose_audio,
            "Generate Demo": self.generate_demo_audio,
        }
        self.num_inputs = 0
        self.num_outputs = 1

    def choose_audio(self):
        path = filedialog.askopenfilename(
            title="WAV-Datei waehlen",
            filetypes=[("WAV", "*.wav")],
        )
        if path:
            self.parameters["pfad"]["value"] = path
            self.load_audio()

    def generate_demo_audio(self):
        base = Path(__file__).resolve().parents[1]
        demo_path = base / "demo_audio.wav"
        self.parameters["pfad"]["value"] = create_demo_audio(str(demo_path))
        self.load_audio()

    def load_audio(self):
        path = self.get("pfad", "")
        if not path:
            self.last_data = None
            return None
        try:
            self.last_data = load_wav(path)
            return self.last_data
        except Exception:
            self.last_data = None
            return None

    def process(self, data):
        return self.load_audio()
