# Audio Pipe-and-filter Demo

Dieses Projekt uebertraegt die in der Vorlesung gezeigte Bildverarbeitung auf die Domaene **Audioverarbeitung**. Die Struktur ist bewusst nah an der Vorlage gehalten: `filters/`, `pipeline/graph.py`, `ui.py` und `main.py`.

Zusatzlich gibt es jetzt einen Tab `Dateneditor`, der sich an Tools wie `SpeechDataBuilder` anlehnt:
- mehrere `.wav` Dateien oder ganze Ordner laden
- Bereiche in der Wellenform markieren
- Segmente mit Labels versehen
- alle Ausschnitte gesammelt als `.wav` exportieren
- Metadaten als `metadata.csv` und `metadata.jsonl` speichern

## Start
```
python main.py
```

## Demo-Ablauf

1. `Quelle`, `Pitch`, `Tempo`, `Rauschen`, `Lautstaerke`, `Freq-Filter`, `Zeitshift`, `Echo`, `Reverb`, `Normalisieren`, `Senke` nach Bedarf auf die Zeichenflaeche legen.
2. Die Knoten von links nach rechts verbinden.
3. In der `Quelle` auf `Generate Demo` klicken oder eine eigene `.wav` laden.
4. Parameter anpassen, dann `Ausfuehren`.
5. Die `Senke` doppelklicken, um Wellenform, Inspector und Spektrum anzuzeigen.

## Dateneditor

1. Zum Tab `Dateneditor` wechseln.
2. `Dateien laden` oder `Ordner laden` verwenden.
3. Bereich in der Wellenform mit der Maus aufziehen oder Start/Ende rechts eingeben.
4. Label eintragen und `Segment merken` klicken.
5. Alle Segmente ueber `Alle Segmente exportieren` speichern.
6. Die Clips landen im Exportordner unter `clips/`, die Metadaten daneben.


## Projektstruktur

```text
audio_pipes_and_filter/
|- filters/
|  |- base_filter.py
|  |- source_filter.py
|  |- gain_filter.py
|  |- pitch_shift_filter.py
|  |- tempo_filter.py
|  |- noise_filter.py
|  |- frequency_filter.py
|  |- time_shift_filter.py
|  |- echo_filter.py
|  |- reverb_filter.py
|  |- normalize_filter.py
|  |- sink_filter.py
|- pipeline/
|  |- graph.py
|- utils/
|  |- audio_utils.py
|- main.py
|- ui.py
|- requirements.txt
```
