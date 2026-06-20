# ESP32 Sprachsteuerung

Firmware für einen ESP32, der Sprachbefehle erkennt und per CAN-Bus weiterleitet.

## Ablauf

1. Benutzer drückt den Taster (GPIO 39, aktiv low).
2. Das Mikrofon (SPH0645 über I2S) nimmt **2 Sekunden** bei 16 kHz auf.
3. Das Audio wird vorverarbeitet (Pre-Emphasis, Normalisierung, Stille-Trim).
4. Ein **64×124 Mel-Spektrogramm** wird berechnet (muss mit `export_esp32_model.py` übereinstimmen).
5. Ein **int8 TFLite-CNN** klassifiziert `licht_an` oder `licht_aus`.
6. Bei ausreichender Konfidenz wird ein **CAN-Frame** (ID `0x100`) gesendet.

## Hardware

| Signal | GPIO |
|--------|------|
| I2S DOUT | 21 |
| I2S BCLK | 17 |
| I2S LRCL | 18 |
| CAN TX | 32 |
| CAN RX | 25 |
| Taster | 39 |

## CAN-Protokoll

| Byte 0 (Befehl) | Bedeutung |
|-----------------|-----------|
| `0x01` | licht_an |
| `0x02` | licht_aus |

Byte 1 enthält die Konfidenz in Prozent (0–100).

## Modul-Übersicht

Siehe die Doxygen-Gruppen:

- @ref audio — Vorverarbeitung und Mel-Spektrogramm
- @ref i2s — Mikrofon-Aufnahme
- @ref ml — TensorFlow Lite Inferenz
- @ref can — TWAI/CAN-Ausgabe
- @ref ui — Taster-Eingabe
- @ref app — Hauptschleife und Einstieg

## Dokumentation erzeugen

```bash
cd ESP_Code/ESP_Code
doxygen docs/Doxyfile
# Ausgabe: docs/html/index.html
```
