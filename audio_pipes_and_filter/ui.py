# -*- coding: utf-8 -*-
"""
UI-Komponenten:
- PipelineUI: Canvas zur Definition der Pipes & Filters, Eigenschaftsspalte rechts,
  Palette zum Hinzufuegen von Knoten, Verbindungsziehen per Maus.
- ViewerUI: Audioanzeige fuer Senken-Daten.
- DataBuilderUI: Schneiden und Labeln von WAV-Dateien fuer Datensaetze.
"""

from __future__ import annotations

import csv
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from filters.echo_filter import EchoFilter
from filters.frequency_filter import FrequencyFilter
from filters.gain_filter import VolumeFilter
from filters.noise_filter import NoiseFilter
from filters.normalize_filter import NormalizeFilter
from filters.pitch_shift_filter import PitchShiftFilter
from filters.reverb_filter import ReverbFilter
from filters.sink_filter import SinkFilter
from filters.source_filter import SourceFilter
from filters.tempo_filter import TempoFilter
from filters.time_shift_filter import TimeShiftFilter
from pipeline.graph import PipelineGraph
from utils.audio_utils import (
    audio_duration,
    audio_peak,
    audio_rms,
    clone_audio,
    load_wav,
    mono_mix,
    save_wav,
)

NODE_BG = "#f7f7f7"
NODE_BORDER = "#666"
NODE_SELECTED = "#4f8ef7"
PORT_FILL_OUT = "#3aa655"
PORT_FILL_IN = "#f0a202"
TEXT_COLOR = "#333"

NODE_WIDTH = 170
NODE_HEIGHT = 64
PORT_RADIUS = 6
REUSABLE_SEGMENT_LABELS = (
    "Blinker Links",
    "Blinker Rechts",
    "Licht an",
    "Licht aus",
    "Innenbeleuchtung an",
    "Innenbeleuchtung aus",
)


class PipelineUI:
    """Pipeline-Editor mit Canvas und Eigenschaftsspalte."""

    def __init__(self, master: tk.Widget, viewer_ui, on_open_view_tab):
        self.master = master
        self.viewer_ui = viewer_ui
        self.on_open_view_tab = on_open_view_tab

        self.graph = PipelineGraph()
        self.node_items = {}
        self.port_items = {}
        self.connection_items = {}
        self.temp_line = None
        self.dragging_node = None
        self.drag_offset = (0, 0)
        self.connecting_from = None
        self.selected_node_id = None
        self.prop_vars = {}

        self.left = ttk.Frame(master)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right = ttk.Frame(master, width=280)
        self.right.pack(side=tk.RIGHT, fill=tk.Y)
        self.right.pack_propagate(False)

        toolbar = ttk.Frame(self.left)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="Palette:").pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Quelle", command=lambda: self.add_node(SourceFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Pitch", command=lambda: self.add_node(PitchShiftFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Tempo", command=lambda: self.add_node(TempoFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Rauschen", command=lambda: self.add_node(NoiseFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Lautstaerke", command=lambda: self.add_node(VolumeFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Freq-Filter", command=lambda: self.add_node(FrequencyFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Zeitshift", command=lambda: self.add_node(TimeShiftFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Echo", command=lambda: self.add_node(EchoFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reverb", command=lambda: self.add_node(ReverbFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Normalize", command=lambda: self.add_node(NormalizeFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Senke", command=lambda: self.add_node(SinkFilter)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Auswahl loeschen", command=self.delete_selected_node).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Ausfuehren", command=self.run_pipeline).pack(side=tk.RIGHT, padx=8)

        self.canvas = tk.Canvas(self.left, bg="#ffffff")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind_all("<Delete>", self.on_delete_key)

        self.prop_title = ttk.Label(self.right, text="Eigenschaften", font=("TkDefaultFont", 10, "bold"))
        self.prop_title.pack(anchor=tk.W, pady=(8, 4), padx=8)
        self.prop_container = ttk.Frame(self.right)
        self.prop_container.pack(fill=tk.Y, expand=True, padx=8)

        hint = ttk.Label(
            self.right,
            text=(
                "Hinweise:\n"
                "- Knoten ueber die Palette hinzufuegen.\n"
                "- Knoten per Drag & Drop verschieben.\n"
                "- Verbindung vom Ausgang (rechts) zum Eingang (links) ziehen.\n"
                "- Ein Ausgang kann auf mehrere Pfade verzweigen.\n"
                "- Quelle doppelklicken oder im Eigenschaftenbereich eine WAV-Datei waehlen.\n"
                "- Jede Senke kann einen eigenen Speicherpfad bekommen.\n"
                "- Senke doppelklicken, um das Ergebnis im Tab 'Anzeige' zu sehen."
            ),
            justify=tk.LEFT,
        )
        hint.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

    def add_node(self, node_cls):
        node = self.graph.add_node(node_cls())
        x = 100 + (len(self.graph.nodes) - 1) * 24
        y = 120 + (len(self.graph.nodes) - 1) * 24
        self.graph.nodes[node.id].pos = (x, y)
        self.draw_node(node.id)

    def draw_node(self, node_id):
        node = self.graph.nodes[node_id]
        x, y = node.pos
        w, h = NODE_WIDTH, NODE_HEIGHT

        rect = self.canvas.create_rectangle(x, y, x + w, y + h, fill=NODE_BG, outline=NODE_BORDER, width=2)
        title = self.canvas.create_text(x + 8, y + 8, anchor=tk.NW, fill=TEXT_COLOR, text=node.display_name())

        self.node_items[rect] = node_id
        self.node_items[title] = node_id

        if not isinstance(node, SourceFilter):
            in_port = self.canvas.create_oval(
                x - PORT_RADIUS,
                y + h / 2 - PORT_RADIUS,
                x + PORT_RADIUS,
                y + h / 2 + PORT_RADIUS,
                fill=PORT_FILL_IN,
                outline="",
            )
            self.port_items[in_port] = (node_id, "in", 0)

        if not isinstance(node, SinkFilter):
            out_port = self.canvas.create_oval(
                x + w - PORT_RADIUS,
                y + h / 2 - PORT_RADIUS,
                x + w + PORT_RADIUS,
                y + h / 2 + PORT_RADIUS,
                fill=PORT_FILL_OUT,
                outline="",
            )
            self.port_items[out_port] = (node_id, "out", 0)

    def redraw(self):
        self.canvas.delete("all")
        self.node_items.clear()
        self.port_items.clear()
        self.connection_items.clear()
        for nid in self.graph.nodes:
            self.draw_node(nid)
        for src, dst in self.graph.edges:
            self.draw_connection(src, dst)
        if self.selected_node_id is not None:
            self.highlight_node(self.selected_node_id)

    def draw_connection(self, src_id, dst_id):
        src = self.graph.nodes[src_id]
        dst = self.graph.nodes[dst_id]
        sx, sy = src.pos
        dx, dy = dst.pos
        x1 = sx + NODE_WIDTH
        y1 = sy + NODE_HEIGHT / 2
        x2 = dx
        y2 = dy + NODE_HEIGHT / 2
        line = self.canvas.create_line(x1, y1, x2, y2, fill="#333", width=2, arrow=tk.LAST)
        self.connection_items[line] = (src_id, dst_id)

    def find_item(self, event, must_be_port=False):
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        if must_be_port:
            for item in items:
                if item in self.port_items:
                    return item
        return items[-1] if items else None

    def on_mouse_down(self, event):
        item = self.find_item(event)
        if item in self.node_items:
            nid = self.node_items[item]
            node = self.graph.nodes[nid]
            self.selected_node_id = nid
            self.highlight_node(nid)
            x, y = node.pos
            self.dragging_node = nid
            self.drag_offset = (event.x - x, event.y - y)
            self.show_properties(node)
        elif item in self.port_items:
            nid, kind, _ = self.port_items[item]
            if kind == "out":
                self.connecting_from = (nid, 0)
                self.temp_line = self.canvas.create_line(
                    event.x,
                    event.y,
                    event.x,
                    event.y,
                    fill="#888",
                    width=2,
                    dash=(4, 2),
                )
        else:
            self.selected_node_id = None
            self.clear_highlight()
            self.clear_properties()

    def on_mouse_drag(self, event):
        if self.dragging_node is not None:
            nid = self.dragging_node
            dx, dy = self.drag_offset
            new_x = event.x - dx
            new_y = event.y - dy
            self.graph.nodes[nid].pos = (new_x, new_y)
            self.redraw()
        elif self.temp_line is not None:
            x1, y1, _, _ = self.canvas.coords(self.temp_line)
            self.canvas.coords(self.temp_line, x1, y1, event.x, event.y)

    def on_mouse_up(self, event):
        if self.dragging_node is not None:
            self.dragging_node = None
        elif self.temp_line is not None and self.connecting_from is not None:
            item = self.find_item(event, True)
            if item in self.port_items:
                nid, kind, _ = self.port_items[item]
                if kind == "in":
                    src_id, _ = self.connecting_from
                    if src_id != nid:
                        try:
                            self.graph.add_edge(src_id, nid)
                            self.draw_connection(src_id, nid)
                        except ValueError as exc:
                            messagebox.showerror("Verbindungsfehler", str(exc))
            self.canvas.delete(self.temp_line)
            self.temp_line = None
            self.connecting_from = None

    def on_double_click(self, event):
        item = self.find_item(event)
        if item not in self.node_items:
            return

        nid = self.node_items[item]
        node = self.graph.nodes[nid]
        if isinstance(node, SourceFilter) and node.last_data is None:
            node.choose_audio()
        if isinstance(node, SinkFilter) and node.last_data is None:
            self.run_pipeline()

        if isinstance(node, (SourceFilter, SinkFilter)):
            data = node.last_data
            if data is None:
                messagebox.showinfo("Anzeige", "Noch keine Audiodaten vorhanden.")
                return
            self.on_open_view_tab()
            self.viewer_ui.show_audio(data)

    def on_delete_key(self, event):
        self.delete_selected_node()

    def highlight_node(self, node_id):
        self.clear_highlight()
        for item, nid in self.node_items.items():
            if nid == node_id and self.canvas.type(item) == "rectangle":
                self.canvas.itemconfigure(item, outline=NODE_SELECTED)

    def clear_highlight(self):
        for item in self.node_items:
            if self.canvas.type(item) == "rectangle":
                self.canvas.itemconfigure(item, outline=NODE_BORDER)

    def clear_properties(self):
        for widget in self.prop_container.winfo_children():
            widget.destroy()
        self.prop_vars = {}

    def delete_selected_node(self):
        if self.selected_node_id is None:
            messagebox.showinfo("Loeschen", "Bitte zuerst einen Knoten auswaehlen.")
            return

        node_id = self.selected_node_id
        self.graph.remove_node(node_id)
        self.selected_node_id = None
        self.clear_properties()
        self.clear_highlight()
        self.redraw()

    def show_properties(self, node):
        self.clear_properties()
        ttk.Label(self.prop_container, text=node.display_name(), font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))

        for key, meta in node.parameters.items():
            row = ttk.Frame(self.prop_container)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=key).pack(side=tk.LEFT)

            var = tk.StringVar(value=str(meta["value"]))
            entry = ttk.Entry(row, textvariable=var, width=20)
            entry.pack(side=tk.RIGHT)
            self.prop_vars[key] = (var, meta)

        for key, fct_ptr in node.functions.items():
            row = ttk.Frame(self.prop_container)
            row.pack(fill=tk.X, pady=2)
            ttk.Button(row, text=key, command=lambda fn=fct_ptr, n=node: self._run_node_function(fn, n)).pack(side=tk.RIGHT)

        ttk.Button(
            self.prop_container,
            text="Parameter uebernehmen",
            command=lambda: self.apply_properties(node),
        ).pack(anchor=tk.E, pady=8)

    def _run_node_function(self, fn, node):
        fn()
        self.show_properties(node)

    def apply_properties(self, node):
        for key, (var, meta) in self.prop_vars.items():
            text = var.get()
            typ = meta.get("type", str)
            try:
                if typ == int:
                    meta["value"] = int(float(text))
                elif typ == float:
                    meta["value"] = float(text)
                elif typ == bool:
                    meta["value"] = text.strip().lower() in ("1", "true", "ja", "yes")
                else:
                    meta["value"] = text
            except Exception:
                messagebox.showerror("Eingabefehler", f"Parameter '{key}' erwartet Typ {typ.__name__}.")
                return
        self.redraw()
        messagebox.showinfo("Eigenschaften", "Parameter aktualisiert.")

    def run_pipeline(self):
        try:
            order = self.graph.topological_order()
        except ValueError as exc:
            messagebox.showerror("Pipelinefehler", str(exc))
            return

        if not order:
            messagebox.showinfo("Pipeline", "Keine Knoten vorhanden.")
            return

        outputs = {}
        for nid in order:
            node = self.graph.nodes[nid]
            preds = self.graph.predecessors(nid)
            inputs = [outputs.get(pred) for pred in preds]
            if len(inputs) > 1:
                out_data = node.process_list(inputs)
            else:
                in_data = inputs[0] if inputs else None
                out_data = node.process(in_data)
            outputs[nid] = out_data

        sinks = [node for node in self.graph.nodes.values() if isinstance(node, SinkFilter) and node.last_data is not None]
        if sinks:
            self.viewer_ui.show_audio(sinks[-1].last_data)
        sink_count = len(sinks)
        message = "Pipeline ausgefuehrt."
        if sink_count:
            message += f"\nAktive Senken mit Ergebnis: {sink_count}"
        messagebox.showinfo("Pipeline", message)


class ViewerUI:
    """Tab zur Audioanzeige (Senken-Ergebnis)."""

    def __init__(self, master: tk.Widget):
        self.master = master
        self.last_audio = None

        self.nb = ttk.Notebook(master)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_waveform = ttk.Frame(self.nb)
        self.nb.add(self.tab_waveform, text="Wellenform")

        self.tab_inspector = ttk.Frame(self.nb)
        self.nb.add(self.tab_inspector, text="Inspector")

        self.tab_spectrum = ttk.Frame(self.nb)
        self.nb.add(self.tab_spectrum, text="Spektrum")

        self.wave_fig = Figure(figsize=(7, 4), dpi=100)
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.tab_waveform)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self.wave_status = ttk.Label(self.tab_waveform, text="t=- | Amplitude=-", anchor="w")
        self.wave_status.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.wave_canvas.mpl_connect("motion_notify_event", self._on_wave_hover)

        info_frame = ttk.Frame(self.tab_inspector)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        self.info_text = tk.StringVar(value="Keine Audiodaten")
        ttk.Label(info_frame, textvariable=self.info_text, justify=tk.LEFT).pack(anchor="w")
        ttk.Button(info_frame, text="WAV speichern", command=self.export_audio).pack(anchor="e", pady=(8, 0))

        self.spectrum_fig = Figure(figsize=(7, 4), dpi=100)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=self.tab_spectrum)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._wave_times = None
        self._wave_values = None

    def show_audio(self, audio_data):
        self.last_audio = audio_data
        if audio_data is None:
            self.info_text.set("Keine Audiodaten")
            self.wave_ax.clear()
            self.spectrum_ax.clear()
            self.wave_canvas.draw()
            self.spectrum_canvas.draw()
            return

        mono = mono_mix(audio_data)
        sample_rate = audio_data["sample_rate"]
        total = len(mono)
        times = np.arange(total, dtype=np.float32) / float(sample_rate)

        self._wave_times = times
        self._wave_values = mono

        self.wave_ax.clear()
        self.wave_ax.plot(times, mono, color="#2563eb", linewidth=0.8)
        self.wave_ax.set_title("Wellenform")
        self.wave_ax.set_xlabel("Zeit (s)")
        self.wave_ax.set_ylabel("Amplitude")
        self.wave_ax.set_ylim(-1.05, 1.05)
        self.wave_ax.grid(True, alpha=0.25)
        self.wave_fig.tight_layout()
        self.wave_canvas.draw()

        fft_values = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(total, d=1.0 / sample_rate)
        magnitudes = np.abs(fft_values)
        self.spectrum_ax.clear()
        self.spectrum_ax.plot(freqs, magnitudes, color="#dc2626", linewidth=0.8)
        self.spectrum_ax.set_title("Frequenzspektrum")
        self.spectrum_ax.set_xlabel("Frequenz (Hz)")
        self.spectrum_ax.set_ylabel("Betrag")
        self.spectrum_ax.set_xlim(0, min(5000, sample_rate / 2))
        self.spectrum_ax.grid(True, alpha=0.25)
        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()

        info = (
            f"Datei: {audio_data.get('name', '-')} | "
            f"Samplerate: {sample_rate} Hz | "
            f"Kanaele: {audio_data['channels']} | "
            f"Dauer: {audio_duration(audio_data):.2f} s | "
            f"Peak: {audio_peak(audio_data):.3f} | "
            f"RMS: {audio_rms(audio_data):.3f}"
        )
        self.info_text.set(info)

    def _on_wave_hover(self, event):
        if self._wave_times is None or self._wave_values is None or event.xdata is None:
            self.wave_status.configure(text="t=- | Amplitude=-")
            return

        idx = int(np.argmin(np.abs(self._wave_times - event.xdata)))
        if 0 <= idx < len(self._wave_values):
            self.wave_status.configure(text=f"t={self._wave_times[idx]:.4f} s | Amplitude={self._wave_values[idx]:.4f}")
        else:
            self.wave_status.configure(text="t=- | Amplitude=-")

    def export_audio(self):
        if self.last_audio is None:
            messagebox.showinfo("Export", "Keine Audiodaten zum Speichern vorhanden.")
            return

        initial_name = Path(self.last_audio.get("name", "output.wav")).stem + "_export.wav"
        path = filedialog.asksaveasfilename(
            title="Ergebnis speichern",
            defaultextension=".wav",
            initialfile=initial_name,
            filetypes=[("WAV", "*.wav")],
        )
        if not path:
            return

        save_wav(self.last_audio, path)
        messagebox.showinfo("Export", f"Datei gespeichert:\n{path}")


class DataBuilderUI:
    """Schneiden und Labeln von WAV-Dateien fuer Datensatz-Erstellung."""

    def __init__(self, master: tk.Widget):
        self.master = master
        self.audio_files = []
        self.audio_by_path = {}
        self.segments_by_source = {}
        self.current_audio = None
        self.current_path = None
        self.current_selection = None
        self._wave_times = None
        self._wave_values = None
        self._selection_patches = []

        self.start_var = tk.StringVar(value="0.000")
        self.end_var = tk.StringVar(value="0.000")
        self.label_var = tk.StringVar(value=REUSABLE_SEGMENT_LABELS[0])
        self.export_dir_var = tk.StringVar(value=str(Path.cwd() / "exports"))
        self.status_var = tk.StringVar(value="Noch keine WAV-Datei geladen.")
        self.file_info_var = tk.StringVar(value="Datei: -")

        self.left = ttk.Frame(master, width=280)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 6), pady=10)
        self.left.pack_propagate(False)

        self.center = ttk.Frame(master)
        self.center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=10)

        self.right = ttk.Frame(master, width=360)
        self.right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 10), pady=10)
        self.right.pack_propagate(False)

        self._build_file_panel()
        self._build_wave_panel()
        self._build_editor_panel()

    def _build_file_panel(self):
        ttk.Label(self.left, text="WAV-Dateien", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)

        file_buttons = ttk.Frame(self.left)
        file_buttons.pack(fill=tk.X, pady=(6, 8))
        ttk.Button(file_buttons, text="Dateien laden", command=self.load_files).pack(fill=tk.X, pady=2)
        ttk.Button(file_buttons, text="Ordner laden", command=self.load_folder).pack(fill=tk.X, pady=2)
        ttk.Button(file_buttons, text="Liste leeren", command=self.clear_files).pack(fill=tk.X, pady=2)

        self.file_listbox = tk.Listbox(self.left, exportselection=False)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        nav = ttk.Frame(self.left)
        nav.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(nav, text="Vorherige", command=lambda: self.step_file(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(nav, text="Naechste", command=lambda: self.step_file(1)).pack(side=tk.LEFT, expand=True, fill=tk.X)

    def _build_wave_panel(self):
        header = ttk.Frame(self.center)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Wellenform", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.file_info_var).pack(side=tk.RIGHT)

        self.wave_fig = Figure(figsize=(8, 4.4), dpi=100)
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.center)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 8))
        self.wave_canvas.mpl_connect("motion_notify_event", self._on_wave_hover)

        self.span_selector = SpanSelector(
            self.wave_ax,
            self.on_span_selected,
            "horizontal",
            useblit=True,
            props={"alpha": 0.2, "facecolor": "#f59e0b"},
            interactive=True,
            drag_from_anywhere=True,
        )

        self.hover_var = tk.StringVar(value="t=- | Amplitude=-")
        ttk.Label(self.center, textvariable=self.hover_var, anchor="w").pack(fill=tk.X)
        ttk.Label(
            self.center,
            text="Bereich mit der Maus aufziehen oder Start/Ende rechts eintragen.",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

    def _build_editor_panel(self):
        ttk.Label(self.right, text="Segment-Editor", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)

        selection_box = ttk.LabelFrame(self.right, text="Auswahl")
        selection_box.pack(fill=tk.X, pady=(8, 8))

        start_row = ttk.Frame(selection_box)
        start_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(start_row, text="Start (s)").pack(side=tk.LEFT)
        ttk.Entry(start_row, textvariable=self.start_var, width=12).pack(side=tk.RIGHT)

        end_row = ttk.Frame(selection_box)
        end_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(end_row, text="Ende (s)").pack(side=tk.LEFT)
        ttk.Entry(end_row, textvariable=self.end_var, width=12).pack(side=tk.RIGHT)

        label_row = ttk.Frame(selection_box)
        label_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(label_row, text="Label").pack(side=tk.LEFT)
        self.label_combo = ttk.Combobox(
            label_row,
            textvariable=self.label_var,
            values=REUSABLE_SEGMENT_LABELS,
            state="readonly",
            width=24,
        )
        self.label_combo.pack(side=tk.RIGHT)

        button_row = ttk.Frame(selection_box)
        button_row.pack(fill=tk.X, padx=8, pady=(6, 8))
        ttk.Button(button_row, text="Auswahl anwenden", command=self.apply_selection_from_entries).pack(fill=tk.X, pady=2)
        ttk.Button(button_row, text="Segment merken", command=self.add_segment).pack(fill=tk.X, pady=2)
        ttk.Button(button_row, text="Auswahl zuruecksetzen", command=self.reset_selection).pack(fill=tk.X, pady=2)

        export_box = ttk.LabelFrame(self.right, text="Export")
        export_box.pack(fill=tk.X, pady=(0, 8))
        export_row = ttk.Frame(export_box)
        export_row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Entry(export_row, textvariable=self.export_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(export_row, text="...", width=4, command=self.choose_export_dir).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(export_box, text="Alle Segmente exportieren", command=self.export_segments).pack(fill=tk.X, padx=8, pady=(4, 8))

        segment_box = ttk.LabelFrame(self.right, text="Gemerkte Segmente")
        segment_box.pack(fill=tk.BOTH, expand=True)

        columns = ("start", "end", "label")
        self.segment_tree = ttk.Treeview(segment_box, columns=columns, show="headings", height=12)
        self.segment_tree.heading("start", text="Start")
        self.segment_tree.heading("end", text="Ende")
        self.segment_tree.heading("label", text="Label")
        self.segment_tree.column("start", width=70, anchor=tk.E)
        self.segment_tree.column("end", width=70, anchor=tk.E)
        self.segment_tree.column("label", width=150, anchor=tk.W)
        self.segment_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self.segment_tree.bind("<<TreeviewSelect>>", self.on_segment_select)

        segment_buttons = ttk.Frame(segment_box)
        segment_buttons.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(segment_buttons, text="Segment loeschen", command=self.delete_selected_segment).pack(fill=tk.X)

        ttk.Label(self.right, textvariable=self.status_var, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=(8, 0))

    def load_files(self):
        paths = filedialog.askopenfilenames(
            title="WAV-Dateien waehlen",
            filetypes=[("WAV", "*.wav")],
        )
        if paths:
            self._register_paths(paths)

    def load_folder(self):
        folder = filedialog.askdirectory(title="Ordner mit WAV-Dateien waehlen")
        if not folder:
            return
        paths = sorted(str(path) for path in Path(folder).glob("*.wav"))
        if not paths:
            messagebox.showinfo("Ordner", "Im ausgewaehlten Ordner wurden keine WAV-Dateien gefunden.")
            return
        self._register_paths(paths)

    def _register_paths(self, paths):
        added = 0
        for raw_path in paths:
            path = str(Path(raw_path).resolve())
            if path in self.audio_by_path:
                continue
            try:
                audio = load_wav(path)
            except Exception as exc:
                messagebox.showerror("Ladefehler", f"Datei konnte nicht geladen werden:\n{path}\n\n{exc}")
                continue
            self.audio_by_path[path] = audio
            self.audio_files.append(path)
            self.segments_by_source.setdefault(path, [])
            self.file_listbox.insert(tk.END, Path(path).name)
            added += 1

        if added and self.current_path is None:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(0)
            self.on_file_select()

        self.status_var.set(f"{len(self.audio_files)} Datei(en) in der Liste.")

    def clear_files(self):
        self.audio_files.clear()
        self.audio_by_path.clear()
        self.segments_by_source.clear()
        self.current_audio = None
        self.current_path = None
        self.current_selection = None
        self.file_listbox.delete(0, tk.END)
        for item in self.segment_tree.get_children():
            self.segment_tree.delete(item)
        self._clear_waveform()
        self.file_info_var.set("Datei: -")
        self.status_var.set("Dateiliste geleert.")
        self.start_var.set("0.000")
        self.end_var.set("0.000")
        self.label_var.set(REUSABLE_SEGMENT_LABELS[0])

    def step_file(self, delta):
        if not self.audio_files:
            return
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
        else:
            index = 0
        new_index = max(0, min(len(self.audio_files) - 1, index + delta))
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(new_index)
        self.file_listbox.see(new_index)
        self.on_file_select()

    def on_file_select(self, event=None):
        selection = self.file_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        path = self.audio_files[index]
        self.current_path = path
        self.current_audio = self.audio_by_path[path]
        duration = audio_duration(self.current_audio)
        self.current_selection = (0.0, duration)
        self.start_var.set("0.000")
        self.end_var.set(f"{duration:.3f}")
        self.label_var.set(REUSABLE_SEGMENT_LABELS[0])
        self.file_info_var.set(
            f"Datei: {Path(path).name} | {duration:.2f}s | {self.current_audio['sample_rate']} Hz | {self.current_audio['channels']} Kanal/Kanaele"
        )
        self._draw_waveform()
        self.refresh_segment_tree()
        self.status_var.set(f"Aktive Datei: {Path(path).name}")

    def _draw_waveform(self):
        if self.current_audio is None:
            self._clear_waveform()
            return

        mono = mono_mix(self.current_audio)
        sample_rate = self.current_audio["sample_rate"]
        self._wave_times = np.arange(len(mono), dtype=np.float32) / float(sample_rate)
        self._wave_values = mono

        self.wave_ax.clear()
        self.wave_ax.plot(self._wave_times, mono, color="#0f766e", linewidth=0.7)
        self.wave_ax.set_title(Path(self.current_path).name)
        self.wave_ax.set_xlabel("Zeit (s)")
        self.wave_ax.set_ylabel("Amplitude")
        self.wave_ax.set_ylim(-1.05, 1.05)
        self.wave_ax.grid(True, alpha=0.25)
        self._render_selection()
        self.wave_fig.tight_layout()
        self.wave_canvas.draw_idle()

    def _clear_waveform(self):
        self._wave_times = None
        self._wave_values = None
        self.wave_ax.clear()
        self.wave_ax.set_title("Keine Datei geladen")
        self.wave_canvas.draw_idle()
        self.hover_var.set("t=- | Amplitude=-")

    def _render_selection(self):
        for patch in self._selection_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self._selection_patches = []

        if self.current_selection is None:
            return

        start, end = self.current_selection
        self._selection_patches.append(self.wave_ax.axvspan(start, end, color="#f59e0b", alpha=0.18))
        self._selection_patches.append(self.wave_ax.axvline(start, color="#d97706", linewidth=1.2))
        self._selection_patches.append(self.wave_ax.axvline(end, color="#d97706", linewidth=1.2))

    def on_span_selected(self, xmin, xmax):
        if self.current_audio is None:
            return
        start, end = sorted((float(xmin), float(xmax)))
        duration = audio_duration(self.current_audio)
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start <= 0.001:
            return
        self.current_selection = (start, end)
        self.start_var.set(f"{start:.3f}")
        self.end_var.set(f"{end:.3f}")
        self._draw_waveform()

    def _on_wave_hover(self, event):
        if self._wave_times is None or self._wave_values is None or event.xdata is None:
            self.hover_var.set("t=- | Amplitude=-")
            return

        idx = int(np.argmin(np.abs(self._wave_times - event.xdata)))
        if 0 <= idx < len(self._wave_values):
            self.hover_var.set(f"t={self._wave_times[idx]:.4f} s | Amplitude={self._wave_values[idx]:.4f}")
        else:
            self.hover_var.set("t=- | Amplitude=-")

    def apply_selection_from_entries(self):
        if self.current_audio is None:
            messagebox.showinfo("Auswahl", "Bitte zuerst eine WAV-Datei laden.")
            return False
        try:
            start = float(self.start_var.get().replace(",", "."))
            end = float(self.end_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Auswahl", "Start und Ende muessen Zahlen sein.")
            return False

        start, end = sorted((start, end))
        duration = audio_duration(self.current_audio)
        if start < 0 or end > duration or end - start <= 0.001:
            messagebox.showerror("Auswahl", f"Ungueltiger Bereich. Erlaubt ist 0 bis {duration:.3f} Sekunden.")
            return False

        self.current_selection = (start, end)
        self._draw_waveform()
        self.status_var.set(f"Auswahl aktualisiert: {start:.3f}s bis {end:.3f}s")
        return True

    def reset_selection(self):
        if self.current_audio is None:
            return
        duration = audio_duration(self.current_audio)
        self.current_selection = (0.0, duration)
        self.start_var.set("0.000")
        self.end_var.set(f"{duration:.3f}")
        self._draw_waveform()

    def add_segment(self):
        if self.current_audio is None or self.current_path is None:
            messagebox.showinfo("Segment", "Bitte zuerst eine WAV-Datei laden.")
            return

        if not self.apply_selection_from_entries():
            return

        label = self.label_var.get().strip()
        if not label:
            messagebox.showerror("Segment", "Bitte ein Label fuer das Segment eintragen.")
            return

        start, end = self.current_selection
        segment = {
            "start": round(start, 3),
            "end": round(end, 3),
            "label": label,
            "source_path": self.current_path,
            "source_name": Path(self.current_path).name,
        }
        self.segments_by_source.setdefault(self.current_path, []).append(segment)
        self.refresh_segment_tree()
        self.label_var.set(REUSABLE_SEGMENT_LABELS[0])
        self.status_var.set(f"Segment gespeichert: {segment['source_name']} [{segment['start']:.3f}s - {segment['end']:.3f}s] -> {label}")

    def refresh_segment_tree(self):
        for item in self.segment_tree.get_children():
            self.segment_tree.delete(item)

        if self.current_path is None:
            return

        for index, segment in enumerate(self.segments_by_source.get(self.current_path, [])):
            self.segment_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(f"{segment['start']:.3f}", f"{segment['end']:.3f}", segment["label"]),
            )

    def on_segment_select(self, event=None):
        if self.current_path is None:
            return
        selection = self.segment_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        segment = self.segments_by_source[self.current_path][index]
        self.start_var.set(f"{segment['start']:.3f}")
        self.end_var.set(f"{segment['end']:.3f}")
        if segment["label"] in REUSABLE_SEGMENT_LABELS:
            self.label_var.set(segment["label"])
        else:
            self.label_var.set(REUSABLE_SEGMENT_LABELS[0])
        self.current_selection = (segment["start"], segment["end"])
        self._draw_waveform()

    def delete_selected_segment(self):
        if self.current_path is None:
            return
        selection = self.segment_tree.selection()
        if not selection:
            messagebox.showinfo("Segment", "Bitte zuerst ein Segment auswaehlen.")
            return
        index = int(selection[0])
        del self.segments_by_source[self.current_path][index]
        self.refresh_segment_tree()
        self.status_var.set("Segment entfernt.")

    def choose_export_dir(self):
        folder = filedialog.askdirectory(title="Exportordner waehlen")
        if folder:
            self.export_dir_var.set(folder)

    def export_segments(self):
        all_segments = []
        for source_path in self.audio_files:
            all_segments.extend(self.segments_by_source.get(source_path, []))

        if not all_segments:
            messagebox.showinfo("Export", "Es sind noch keine Segmente zum Export vorhanden.")
            return

        export_dir = Path(self.export_dir_var.get().strip() or (Path.cwd() / "exports")).resolve()
        clips_dir = export_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        for index, segment in enumerate(all_segments, start=1):
            source_audio = self.audio_by_path[segment["source_path"]]
            sample_rate = source_audio["sample_rate"]
            start_frame = int(segment["start"] * sample_rate)
            end_frame = int(segment["end"] * sample_rate)
            sliced = source_audio["samples"][start_frame:end_frame]
            safe_label = self._slugify(segment["label"]) or "label"
            stem = Path(segment["source_name"]).stem
            clip_name = f"{stem}_{index:04d}_{safe_label}.wav"
            clip_path = clips_dir / clip_name
            clip_audio = clone_audio(
                source_audio,
                samples=sliced,
                name=clip_name,
                path=str(clip_path),
            )
            save_wav(clip_audio, str(clip_path))

            metadata_rows.append(
                {
                    "clip_path": str(clip_path),
                    "clip_name": clip_name,
                    "source_name": segment["source_name"],
                    "source_path": segment["source_path"],
                    "start_seconds": f"{segment['start']:.3f}",
                    "end_seconds": f"{segment['end']:.3f}",
                    "duration_seconds": f"{segment['end'] - segment['start']:.3f}",
                    "label": segment["label"],
                }
            )

        csv_path = export_dir / "metadata.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "clip_path",
                    "clip_name",
                    "source_name",
                    "source_path",
                    "start_seconds",
                    "end_seconds",
                    "duration_seconds",
                    "label",
                ],
            )
            writer.writeheader()
            writer.writerows(metadata_rows)

        jsonl_path = export_dir / "metadata.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in metadata_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.status_var.set(
            f"{len(metadata_rows)} Segment(e) exportiert nach {clips_dir}. Metadaten: {csv_path.name}, {jsonl_path.name}"
        )
        messagebox.showinfo(
            "Export abgeschlossen",
            f"{len(metadata_rows)} Segment(e) wurden exportiert.\n\nOrdner:\n{clips_dir}\n\nMetadaten:\n{csv_path}\n{jsonl_path}",
        )

    @staticmethod
    def _slugify(text):
        sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
        return sanitized.strip("_").lower()
