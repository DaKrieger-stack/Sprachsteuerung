/**
 * @file modell.h
 * @brief Eingebettetes TFLite-Modell für die Sprachklassifikation.
 *
 * Die Byte-Arrays werden von `scripts/export_esp32_model.py` nach
 * `src/modell.c` exportiert. Nicht manuell bearbeiten.
 */

#ifndef MODELL_H
#define MODELL_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Rohes TFLite-FlatBuffer (int8-quantisiertes CNN). */
extern const unsigned char tiny_cnn_model_big_tflite[];

/** @brief Länge von @ref tiny_cnn_model_big_tflite in Bytes. */
extern const unsigned int tiny_cnn_model_big_tflite_len;

#ifdef __cplusplus
}
#endif

#endif
