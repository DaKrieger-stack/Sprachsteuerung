# ESP32 Sprachsteuerung mit I2S-Mikrofon

Diese Anwendung erkennt genau zwei Sprachbefehle auf einem ESP32-WROOM-32E:

- `Licht an`  -> CAN-Frame fuer Licht EIN wird gesendet
- `Licht aus` -> CAN-Frame fuer Licht AUS wird gesendet

## Dateien

- `main.py`: Hauptprogramm mit Listen- und Trainingsmodus
- `audio_capture.py`: 1s I2S-Aufnahme mit 16 kHz Mono
- `preprocess.py`: Log-Mel-Feature-Extraktion (`20 x 49`)
- `classifier.py`: Template-Matching mit Cosine Similarity
- `template_store.py`: Speichern/Laden von Referenztemplates im Flash
- `config.py`: Pins, Schwellwerte und Feature-Parameter

## Verdrahtung

Standardbelegung in `config.py`:

- `BCLK` -> GPIO `17`
- `LRCL/WS` -> GPIO `18`
- `DOUT/SD` -> GPIO `21`
- `CAN TX` -> GPIO `32`
- `CAN RX` -> GPIO `26`

## Starten

Standardbetrieb:

```python
import main
main.main()
```

Trainingsmodus ueber serielle REPL:

```python
import main
main.main(training=True)
```

Verfuegbare Trainingsbefehle:

- `on`: neues Template fuer `Licht an` aufnehmen
- `off`: neues Template fuer `Licht aus` aufnehmen
- `avg_on`: alle `ON`-Templates zu einem Mittelwerttemplate verdichten
- `avg_off`: alle `OFF`-Templates zu einem Mittelwerttemplate verdichten
- `clear_on`
- `clear_off`
- `clear_all`
- `listen`
- `exit`

## Tuning

Wichtige Parameter in `config.py`:

- `MIN_SIGNAL_RMS`: ignoriert zu leise Fenster
- `SIMILARITY_THRESHOLD`: Mindestscore fuer Treffer
- `MARGIN_THRESHOLD`: Mindestabstand zwischen bestem und zweitbestem Kommando
- `MAX_TEMPLATES_PER_CLASS`: maximale Zahl gespeicherter Referenzen je Klasse
- `CAN_ARB_ID_LIGHT`: CAN-Identifier fuer Lichtbefehle
- `CAN_PAYLOAD_LIGHT_ON` und `CAN_PAYLOAD_LIGHT_OFF`: Nutzdaten fuer EIN/AUS

## Hinweise

- Wenn diese Datei auf dem ESP32 als `/main.py` liegt, startet MicroPython den Listenmodus beim Booten automatisch.
- Die Ausgabe verwendet `machine.CAN`. Falls deine eingesetzte MicroPython-Firmware eine leicht andere CAN-API fuer den verwendeten Controller hat, muss nur `output_control.py` entsprechend angepasst werden.
- Viele I2S-MEMS-Mikrofone liefern intern 24-Bit-Daten in 32-Bit-Frames. Deshalb ist `I2S_BITS` standardmaessig auf `32` gesetzt, das Signal wird aber auf 16-Bit-PCM heruntergerechnet.
- Wenn die verwendete MicroPython-Firmware echtes `16`-Bit-I2S-RX fuer dein Modul sauber unterstuetzt, kann `I2S_BITS` in `config.py` auf `16` geaendert werden.
