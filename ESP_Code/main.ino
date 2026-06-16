/*
  ESP32 Audio-Klassifikation mit TensorFlow Lite Micro
  Modell: tiny_cnn_model_big.tflite

  Pinbelegung (laut Schaltplan):
    INMP441 I2S Mikrofon:
      BCLK  -> GPIO17
      LRCL (WS) -> GPIO18
      DOUT  -> GPIO21
      L/R   -> GND (für linken Kanal, je nach Beschaltung)
      VDD   -> 3.3V
      GND   -> GND

  Vorgehen:
    1. tiny_cnn_model_big.tflite in ein C-Array konvertieren:
       xxd -i tiny_cnn_model_big.tflite > model_data.h
       (oder mit dem Python-Tool tflite-micro/python/tflite_micro/util)
    2. model_data.h in den Sketch-Ordner legen.
    3. Bibliothek "Arduino_TensorFlowLite" oder "TensorFlowLite_ESP32" installieren.
*/

#include <driver/i2s.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h" // enthält das Array g_model[] und g_model_len

// ---------- Konfiguration ----------
#define I2S_WS    18   // LRCL
#define I2S_SD    21   // DOUT
#define I2S_SCK   17   // BCLK
#define I2S_PORT  I2S_NUM_0

#define SAMPLE_RATE     16000
#define SAMPLE_BITS     16
#define AUDIO_LEN_MS    1000               // Länge des Audiofensters
#define SAMPLES_PER_WINDOW (SAMPLE_RATE * AUDIO_LEN_MS / 1000)

// Anzahl Klassen, ggf. anpassen
#define NUM_CLASSES 4
const char* kClassLabels[NUM_CLASSES] = {"klasse_0", "klasse_1", "klasse_2", "klasse_3"};

// ---------- TFLite Globals ----------
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 80 * 1024; // ggf. anpassen falls Modell größer/kleiner
  uint8_t tensor_arena[kTensorArenaSize];
}

// Puffer für Audiodaten
int16_t audio_buffer[SAMPLES_PER_WINDOW];

// ---------- I2S Setup ----------
void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // INMP441 liefert effektiv 24 Bit in 32-Bit-Frame
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

// Liest ein Fenster von SAMPLES_PER_WINDOW 16-Bit-Samples ein
void readAudioWindow(int16_t* out, size_t num_samples) {
  size_t bytes_read = 0;
  size_t samples_filled = 0;

  while (samples_filled < num_samples) {
    int32_t raw_sample;
    size_t br;
    i2s_read(I2S_PORT, &raw_sample, sizeof(int32_t), &br, portMAX_DELAY);
    if (br == sizeof(int32_t)) {
      // 32-Bit-Frame -> auf 16 Bit reduzieren (oberste 16 Bit nutzen)
      int16_t sample16 = (int16_t)(raw_sample >> 16);
      out[samples_filled++] = sample16;
    }
  }
}

// ---------- Vorverarbeitung ----------
// Hier werden die rohen int16-Samples in das Eingabeformat
// des Modells gebracht (z.B. normalisiert auf float -1..1).
// Falls dein Modell auf Spektrogrammen / MFCCs trainiert wurde,
// muss diese Funktion entsprechend erweitert werden!
void preprocessAndFill(const int16_t* audio, size_t num_samples, TfLiteTensor* input_tensor) {
  if (input_tensor->type == kTfLiteFloat32) {
    float* in_data = input_tensor->data.f;
    for (size_t i = 0; i < num_samples; i++) {
      in_data[i] = audio[i] / 32768.0f; // Normalisierung auf [-1,1]
    }
  } else if (input_tensor->type == kTfLiteInt8) {
    int8_t* in_data = input_tensor->data.int8;
    float scale = input_tensor->params.scale;
    int32_t zero_point = input_tensor->params.zero_point;
    for (size_t i = 0; i < num_samples; i++) {
      float val = audio[i] / 32768.0f;
      int32_t q = (int32_t)(val / scale) + zero_point;
      if (q > 127) q = 127;
      if (q < -128) q = -128;
      in_data[i] = (int8_t)q;
    }
  }
}

// ---------- TFLite Setup ----------
void setupModel() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Modellversion stimmt nicht mit Schema ueberein!");
    while (1) delay(1000);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() fehlgeschlagen!");
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
}

// ---------- Inferenz ausführen ----------
void runInference() {
  preprocessAndFill(audio_buffer, SAMPLES_PER_WINDOW, input);

  unsigned long start = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long duration = millis() - start;

  if (invoke_status != kTfLiteOk) {
    Serial.println("Fehler bei Invoke()!");
    return;
  }

  // Ergebnis auswerten (Annahme: Softmax-Output mit NUM_CLASSES Werten)
  int best_idx = -1;
  float best_score = -1e9;

  for (int i = 0; i < NUM_CLASSES; i++) {
    float score;
    if (output->type == kTfLiteFloat32) {
      score = output->data.f[i];
    } else { // int8 quantisiert
      score = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    }
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  Serial.printf("Inferenzzeit: %lu ms | Erkannt: %s (Score: %.3f)\n",
                duration, kClassLabels[best_idx], best_score);
}

// ---------- Setup / Loop ----------
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starte Spracherkennung...");

  setupI2S();
  setupModel();

  Serial.println("Initialisierung abgeschlossen. Beginne Aufnahme...");
}

void loop() {
  readAudioWindow(audio_buffer, SAMPLES_PER_WINDOW);
  runInference();
}
