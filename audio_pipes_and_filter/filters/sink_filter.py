# -*- coding: utf-8 -*-
"""
Senke-Filter: Speichert die empfangenen Audiodaten, damit sie im Anzeige-Tab
gezeigt und optional als WAV gespeichert werden koennen.
"""

from pathlib import Path
from tkinter import filedialog

from filters.base_filter import BaseFilter
from utils.audio_utils import save_wav


class SinkFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "name": {"type": str, "value": "Senke"},
            "status": {"type": str, "value": "Kein Signal"},
            "output_path": {"type": str, "value": ""},
        }
        self.functions = {
            "Choose Save Path...": self.choose_output_path,
            "Save Output": self.save_last_output,
        }
        self.num_inputs = 1
        self.num_outputs = 0

    def choose_output_path(self):
        current_path = self.get("output_path", "")
        initial_name = ""
        if current_path:
            initial_name = Path(current_path).name
        elif self.last_data is not None:
            initial_name = Path(self.last_data.get("name", "output.wav")).stem + "_output.wav"
        else:
            initial_name = "output.wav"

        path = filedialog.asksaveasfilename(
            title="Ergebnis fuer diese Senke speichern",
            defaultextension=".wav",
            initialfile=initial_name,
            filetypes=[("WAV", "*.wav")],
        )
        if path:
            self.parameters["output_path"]["value"] = path
        return path

    def save_last_output(self):
        if self.last_data is None:
            self.parameters["status"]["value"] = "Kein Signal zum Speichern"
            return None

        path = self.get("output_path", "")
        if not path:
            path = self.choose_output_path()
            if not path:
                self.parameters["status"]["value"] = "Speichern abgebrochen"
                return None

        save_wav(self.last_data, path)
        self.parameters["output_path"]["value"] = path
        self.parameters["status"]["value"] = f"Gespeichert: {Path(path).name}"
        return path

    def process(self, audio_data):
        self.last_data = audio_data
        if audio_data is None:
            self.parameters["status"]["value"] = "Kein Signal"
            return None

        output_path = self.get("output_path", "").strip()
        if output_path:
            save_wav(audio_data, output_path)
            self.parameters["status"]["value"] = f"Gespeichert: {Path(output_path).name}"
        else:
            self.parameters["status"]["value"] = "Signal vorhanden"
        return audio_data
