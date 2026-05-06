# -*- coding: utf-8 -*-
"""
Hauptstartpunkt fuer die tkinter-Anwendung.
Erstellt ein Notebook mit den Tabs "Pipeline" und "Anzeige".
Die Pipeline-Ansicht enthaelt ein Canvas zum Anordnen von Filter-Knoten
sowie eine Eigenschaftsspalte auf der rechten Seite.
"""

import tkinter as tk
from tkinter import ttk

from ui import DataBuilderUI, PipelineUI, ViewerUI


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pipes & Filters - Audioverarbeitung (tkinter)")
        self.geometry("1280x820")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.pipeline_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pipeline_frame, text="Pipeline")

        self.viewer_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viewer_frame, text="Anzeige")

        self.builder_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.builder_frame, text="Dateneditor")

        self.viewer_ui = ViewerUI(self.viewer_frame)
        self.builder_ui = DataBuilderUI(self.builder_frame)
        self.pipeline_ui = PipelineUI(
            self.pipeline_frame,
            self.viewer_ui,
            on_open_view_tab=self._open_view_tab,
        )

    def _open_view_tab(self):
        self.notebook.select(self.viewer_frame)


if __name__ == "__main__":
    app = App()
    app.mainloop()
