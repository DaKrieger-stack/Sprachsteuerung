// SPH0645 I2S Mikrofon → Sprachklassifikation → CAN
// Pinbelegung (Schaltplan):
//   Mikrofon:
//     DOUT  → GPIO 21
//     BCLK  → GPIO 17
//     LRCL  → GPIO 18
//   CAN:
//     TX    → GPIO 32  (CAN2_TX)
//     RX    → GPIO 25  (CAN2_RX)
//   Taster S21 → GPIO 39 (Pull-up, aktiv low)
//   S2D1: Power-LED an 3V3, nicht GPIO-gesteuert

#include <driver/gpio.h>
#include <driver/i2s.h>
#include <driver/twai.h>
#include <esp_dsp.h>
#include <esp_err.h>
#include <esp_log.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>

static const char *TAG = "MAIN";

class SerialClass {
 public:
  void begin(unsigned long) {}
  void println(const char *msg) { ESP_LOGI(TAG, "%s", msg); }
  void printf(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
    va_list args;
    va_start(args, fmt);
    esp_log_writev(ESP_LOG_INFO, TAG, fmt, args);
    va_end(args);
  }
  operator bool() const { return true; }
};

static SerialClass Serial;

static inline void delay(uint32_t ms) {
  vTaskDelay(pdMS_TO_TICKS(ms));
}

static inline void yieldCpu() {
  vTaskDelay(1);
}

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "modell.h"

// I2S Konfiguration
#define I2S_PORT        I2S_NUM_0
#define I2S_DOUT        21    // Dateneingang vom Mikrofon
#define I2S_BCLK        17    // Bit-Clock
#define I2S_LRCL        18    // Left/Right-Clock (Word Select)

// CAN / Taster
#define CAN_TX_GPIO     32
#define CAN_RX_GPIO     25
#define BUTTON_GPIO     39

// Audio-Parameter
// Audio-Parameter
#define SAMPLE_RATE     16000   // 16 kHz Abtastrate
#define BITS_PER_SAMPLE 32      // SPH0645 liefert 24-bit in 32-bit Frame
#define BUFFER_LEN      512     // Anzahl Samples pro Lesevorgang
#define DURATION        2.0f
#define SR              SAMPLE_RATE
#define SAMPLES_COUNT   ((int)(DURATION * SAMPLE_RATE))

// Modell-Parameter
#define N_MELS          64
#define N_FFT           512
#define HOP_LENGTH      256
#define REFERENCE_WIDTH 124
#define MIN_SIGNAL_RMS  0.01f
#define MIN_CONFIDENCE  0.45f
#define NUM_CLASSES     3

// CAN-Protokoll
#define CAN_ID_VOICE      0x100
#define CAN_CMD_UNKNOWN   0x00
#define CAN_CMD_LICHT_AN  0x01
#define CAN_CMD_LICHT_AUS 0x02

#define BUTTON_DEBOUNCE_MS 50

static bool g_i2s_installed = false;

const char *CLASS_NAMES[NUM_CLASSES] = {"licht_an", "licht_aus", "unknown"};

namespace {
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input_tensor = nullptr;
  constexpr int kTensorArenaSize = 120 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

typedef struct {
  char predicted_label[32];
  float confidence;
  float probabilities[NUM_CLASSES];
  float signal_rms;
  int class_index;
} ClassificationResult;

static float hann_window[N_FFT];
static float fft_buffer[2 * N_FFT];
static int16_t *audio_buffer = nullptr;
static float *model_input_buffer = nullptr;
static float power_spec_buf[N_FFT / 2 + 1];

static inline float int16SampleToFloat(int16_t sample) {
  return (float)sample / 32768.0f;
}

static inline int16_t i2sSampleToInt16(int32_t raw) {
  int32_t scaled = raw >> 16;
  if (scaled > 32767) return 32767;
  if (scaled < -32768) return -32768;
  return (int16_t)scaled;
}

static inline float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static inline float melFilterWeight(int m, int fft_idx, float mel_min, float mel_step) {
  float f = (float)fft_idx * (float)SR / (float)N_FFT;
  float hz_left = mel_to_hz(mel_min + m * mel_step);
  float hz_center = mel_to_hz(mel_min + (m + 1) * mel_step);
  float hz_right = mel_to_hz(mel_min + (m + 2) * mel_step);

  if (f >= hz_left && f <= hz_center) {
    return (f - hz_left) / (hz_center - hz_left);
  }
  if (f >= hz_center && f <= hz_right) {
    return (hz_right - f) / (hz_right - hz_center);
  }
  return 0.0f;
}

float audioRmsInt16(const int16_t *audio, size_t len) {
  if (len == 0) return 0.0f;
  double sum = 0.0;
  for (size_t i = 0; i < len; i++) {
    float sample = int16SampleToFloat(audio[i]);
    sum += (double)sample * (double)sample;
  }
  return (float)sqrt(sum / (double)len);
}

size_t trimSilenceInt16(int16_t *audio, size_t len, float threshold_db) {
  float threshold_linear = powf(10.0f, threshold_db / 20.0f);

  size_t start = 0;
  for (size_t i = 0; i < len; i++) {
    if (fabsf(int16SampleToFloat(audio[i])) > threshold_linear) {
      start = i;
      break;
    }
  }

  size_t end = len;
  for (size_t i = len; i > 0; i--) {
    if (fabsf(int16SampleToFloat(audio[i - 1])) > threshold_linear) {
      end = i;
      break;
    }
  }

  if (end <= start) return len;

  size_t new_len = end - start;
  memmove(audio, audio + start, new_len * sizeof(int16_t));
  return new_len;
}

void prepareAudioClipInPlaceInt16(int16_t *audio, size_t len) {
  const float preemph = 0.97f;
  for (size_t i = len - 1; i > 0; i--) {
    float val = int16SampleToFloat(audio[i]) -
                preemph * int16SampleToFloat(audio[i - 1]);
    if (val > 1.0f) val = 1.0f;
    if (val < -1.0f) val = -1.0f;
    audio[i] = (int16_t)lroundf(val * 32767.0f);
    if ((i & 4095) == 0) {
      yieldCpu();
    }
  }

  int16_t max_val = 0;
  for (size_t i = 0; i < len; i++) {
    int16_t abs_val = audio[i] < 0 ? (int16_t)(-audio[i]) : audio[i];
    if (abs_val > max_val) max_val = abs_val;
  }
  if (max_val > 0) {
    for (size_t i = 0; i < len; i++) {
      audio[i] = (int16_t)(((int32_t)audio[i] * 32767) / max_val);
    }
  }
}

int getMelSpecInt16(const int16_t *audio, size_t audio_len, float *output, int max_frames) {
  static bool window_initialized = false;
  if (!window_initialized) {
    dsps_wind_hann_f32(hann_window, N_FFT);
    window_initialized = true;
  }

  int num_frames = (int)((audio_len - N_FFT) / HOP_LENGTH) + 1;
  if (num_frames <= 0) num_frames = 1;
  if (num_frames > max_frames) num_frames = max_frames;

  float *power_spec = power_spec_buf;
  float global_max = -INFINITY;
  float global_min = INFINITY;

  const float mel_min = hz_to_mel(0.0f);
  const float mel_max = hz_to_mel(SR / 2.0f);
  const float mel_step = (mel_max - mel_min) / (N_MELS + 1);

  for (int frame = 0; frame < num_frames; frame++) {
    int start = frame * HOP_LENGTH;

    for (int i = 0; i < N_FFT; i++) {
      if (start + i < (int)audio_len) {
        fft_buffer[i * 2] =
            int16SampleToFloat(audio[start + i]) * hann_window[i];
      } else {
        fft_buffer[i * 2] = 0.0f;
      }
      fft_buffer[i * 2 + 1] = 0.0f;
    }

    dsps_fft2r_fc32(fft_buffer, N_FFT);
    dsps_bit_rev_fc32(fft_buffer, N_FFT);
    dsps_cplx2reC_fc32(fft_buffer, N_FFT);

    for (int i = 0; i <= N_FFT / 2; i++) {
      float re = fft_buffer[i * 2];
      float im = fft_buffer[i * 2 + 1];
      power_spec[i] = re * re + im * im;
    }

    for (int m = 0; m < N_MELS; m++) {
      float mel_energy = 0.0f;

      for (int fft_idx = 0; fft_idx <= N_FFT / 2; fft_idx++) {
        float weight = melFilterWeight(m, fft_idx, mel_min, mel_step);
        if (weight > 0.0f) {
          mel_energy += weight * power_spec[fft_idx];
        }
      }

      if (mel_energy < 1e-10f) mel_energy = 1e-10f;

      float log_mel = 10.0f * log10f(mel_energy);
      output[m * num_frames + frame] = log_mel;

      if (log_mel > global_max) global_max = log_mel;
      if (log_mel < global_min) global_min = log_mel;
    }

    yieldCpu();
  }

  float range = global_max - global_min;
  if (range < 1e-8f) range = 1e-8f;

  for (int m = 0; m < N_MELS; m++) {
    for (int t = 0; t < num_frames; t++) {
      output[m * num_frames + t] =
          (output[m * num_frames + t] - global_min) / range;
    }
  }

  return num_frames;
}

void padOrCropTime(const float *spec, int current_width, int target_width,
                   int n_mels, float *output) {
  if (current_width > target_width) {
    int start = (current_width - target_width) / 2;
    for (int m = 0; m < n_mels; m++) {
      for (int t = 0; t < target_width; t++) {
        output[m * target_width + t] = spec[m * current_width + start + t];
      }
    }
  } else if (current_width < target_width) {
    for (int m = 0; m < n_mels; m++) {
      for (int t = 0; t < current_width; t++) {
        output[m * target_width + t] = spec[m * current_width + t];
      }
      for (int t = current_width; t < target_width; t++) {
        output[m * target_width + t] = 0.0f;
      }
    }
  } else {
    memcpy(output, spec, n_mels * current_width * sizeof(float));
  }
}

static void discardI2SChunk() {
  int32_t trash[BUFFER_LEN];
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, trash, sizeof(trash), &bytes_read, pdMS_TO_TICKS(50));
}

static int32_t i2sPickSample(int32_t left, int32_t right) {
  int32_t abs_left = left < 0 ? -left : left;
  int32_t abs_right = right < 0 ? -right : right;
  return abs_left >= abs_right ? left : right;
}

static int32_t i2sSampleToRaw(int32_t raw) {
  return raw >> 8;
}

static void processI2SBlock(const int32_t *raw_samples, int samples_read,
                            int16_t *out, size_t *total_samples, size_t sample_count,
                            int32_t *raw_peak) {
  // ESP-IDF 5: Stereo L/R im Buffer; SPH0645 (SEL=GND) -> linker Kanal
  if (samples_read >= 2) {
    for (int i = 0; i + 1 < samples_read && *total_samples < sample_count; i += 2) {
      int32_t left = raw_samples[i];
      int32_t right = raw_samples[i + 1];
      int32_t pick = i2sPickSample(left, right);
      int32_t abs_raw = pick < 0 ? -pick : pick;
      if (abs_raw > *raw_peak) {
        *raw_peak = abs_raw;
      }
      out[(*total_samples)++] = i2sSampleToInt16(pick);
    }
    return;
  }

  for (int i = 0; i < samples_read && *total_samples < sample_count; i++) {
    int32_t abs_raw = raw_samples[i] < 0 ? -raw_samples[i] : raw_samples[i];
    if (abs_raw > *raw_peak) {
      *raw_peak = abs_raw;
    }
    out[(*total_samples)++] = i2sSampleToInt16(raw_samples[i]);
  }
}

static float pollI2SRmsNorm() {
  int32_t raw_samples[BUFFER_LEN];
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, 0);
  if (bytes_read == 0) {
    return 0.0f;
  }

  int samples_read = (int)(bytes_read / sizeof(int32_t));
  double sum = 0.0;
  int count = 0;

  if (samples_read >= 2) {
    for (int i = 0; i + 1 < samples_read; i += 2) {
      int32_t pick = i2sPickSample(raw_samples[i], raw_samples[i + 1]);
      int32_t sample = i2sSampleToRaw(pick);
      sum += (double)sample * (double)sample;
      count++;
    }
  } else {
    for (int i = 0; i < samples_read; i++) {
      int32_t sample = i2sSampleToRaw(raw_samples[i]);
      sum += (double)sample * (double)sample;
      count++;
    }
  }

  if (count <= 0) {
    return 0.0f;
  }
  return (float)(sqrt(sum / count) / 8388607.0);
}

static float pollI2SRmsNormBlocking() {
  int32_t raw_samples[BUFFER_LEN];
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, pdMS_TO_TICKS(200));
  if (bytes_read == 0) {
    return 0.0f;
  }

  int samples_read = (int)(bytes_read / sizeof(int32_t));
  double sum = 0.0;
  int count = 0;

  if (samples_read >= 2) {
    for (int i = 0; i + 1 < samples_read; i += 2) {
      int32_t pick = i2sPickSample(raw_samples[i], raw_samples[i + 1]);
      int32_t sample = i2sSampleToRaw(pick);
      sum += (double)sample * (double)sample;
      count++;
    }
  } else {
    for (int i = 0; i < samples_read; i++) {
      int32_t sample = i2sSampleToRaw(raw_samples[i]);
      sum += (double)sample * (double)sample;
      count++;
    }
  }

  if (count <= 0) {
    return 0.0f;
  }
  return (float)(sqrt(sum / count) / 8388607.0);
}

static esp_err_t startI2SDriver() {
  i2s_config_t i2s_config = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 4,
    .dma_buf_len          = BUFFER_LEN,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };

  i2s_pin_config_t pin_config = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = I2S_BCLK,
    .ws_io_num    = I2S_LRCL,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_DOUT
  };

  esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    return err;
  }

  err = i2s_set_pin(I2S_PORT, &pin_config);
  if (err != ESP_OK) {
    i2s_driver_uninstall(I2S_PORT);
    return err;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  delay(100);
  g_i2s_installed = true;
  return ESP_OK;
}

static void stopI2SDriver() {
  if (!g_i2s_installed) {
    return;
  }
  i2s_driver_uninstall(I2S_PORT);
  g_i2s_installed = false;
}

static void warmupI2S() {
  for (int i = 0; i < 6; i++) {
    discardI2SChunk();
  }
}

static void prepareI2SCapture() {
  if (!g_i2s_installed) {
    esp_err_t err = startI2SDriver();
    if (err != ESP_OK) {
      Serial.printf("Fehler beim I2S-Start fuer Aufnahme: %d\n", err);
      return;
    }
  }
  warmupI2S();
}

void setupI2S() {
  esp_err_t err = startI2SDriver();
  if (err != ESP_OK) {
    Serial.printf("Fehler beim I2S-Setup: %d\n", err);
    while (true) delay(1000);
  }

  Serial.println("I2S Live-Monitor (RMS normiert, wie Arduino-Test):");
  for (int i = 0; i < 5; i++) {
    float rms_norm = pollI2SRmsNorm();
    Serial.printf("  RMS: %.4f\n", rms_norm);
    delay(200);
  }
}

void recordMicrophone(int16_t *out, size_t sample_count) {
  prepareI2SCapture();

  float pre_rms = pollI2SRmsNormBlocking();
  Serial.printf(">> Aufnahme (%.1fs) - jetzt sprechen ... (Pre-RMS=%.4f)\n",
                DURATION, pre_rms);

  size_t total_samples = 0;
  int32_t raw_samples[BUFFER_LEN];
  int32_t raw_peak = 0;

  while (total_samples < sample_count) {
    size_t bytes_read = 0;
    i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);

    int samples_read = (int)(bytes_read / sizeof(int32_t));
    processI2SBlock(raw_samples, samples_read, out, &total_samples, sample_count, &raw_peak);
  }

  float rms = audioRmsInt16(out, sample_count);
  Serial.printf("Aufnahme RMS: %.4f (raw peak=%ld)\n", rms, (long)raw_peak);
}

void setupModel() {
  const tflite::Model *model = tflite::GetModel(tiny_cnn_model_big_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Fehler: Modellversion passt nicht!");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("Fehler: Tensor-Allokierung (Arena %u KB, frei: %u Bytes)\n",
                  (unsigned)(kTensorArenaSize / 1024),
                  (unsigned)esp_get_free_heap_size());
    while (true) delay(1000);
  }

  input_tensor = interpreter->input(0);
  Serial.printf("Modell OK (%u Bytes, ESP32-int8, Arena %u KB)\n",
                tiny_cnn_model_big_tflite_len,
                (unsigned)(kTensorArenaSize / 1024));
  Serial.printf("Input-Typ: %d, Bytes: %u\n", input_tensor->type, input_tensor->bytes);
}

void runInference(const float *input_data, float *output_probs, int num_classes) {
  memcpy(input_tensor->data.f, input_data, input_tensor->bytes);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Fehler: Inference fehlgeschlagen!");
    return;
  }

  TfLiteTensor *output = interpreter->output(0);
  for (int i = 0; i < num_classes; i++) {
    output_probs[i] = output->data.f[i];
  }
}

void classifyAudio(int16_t *audio, size_t len, ClassificationResult *result) {
  if (!audio || !model_input_buffer) {
    Serial.println("Fehler: Audio-Arbeitspuffer nicht initialisiert!");
    return;
  }

  result->signal_rms = audioRmsInt16(audio, len);

  prepareAudioClipInPlaceInt16(audio, len);

  size_t trimmed_len = trimSilenceInt16(audio, len, -40.0f);

  int actual_frames =
      getMelSpecInt16(audio, trimmed_len, model_input_buffer, REFERENCE_WIDTH);
  if (actual_frames == 0) {
    return;
  }

  if (actual_frames != REFERENCE_WIDTH) {
    padOrCropTime(model_input_buffer, actual_frames, REFERENCE_WIDTH, N_MELS, model_input_buffer);
  }

  yieldCpu();
  float probs[NUM_CLASSES];
  runInference(model_input_buffer, probs, NUM_CLASSES);
  yieldCpu();

  int max_idx = 0;
  float max_prob = probs[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      max_idx = i;
    }
  }

  strncpy(result->predicted_label, CLASS_NAMES[max_idx], sizeof(result->predicted_label) - 1);
  result->predicted_label[sizeof(result->predicted_label) - 1] = '\0';
  result->confidence = max_prob;
  result->class_index = max_idx;
  for (int i = 0; i < NUM_CLASSES; i++) {
    result->probabilities[i] = probs[i];
  }
}

void setupCan() {
  twai_general_config_t g_config =
      TWAI_GENERAL_CONFIG_DEFAULT((gpio_num_t)CAN_TX_GPIO, (gpio_num_t)CAN_RX_GPIO,
                                  TWAI_MODE_NORMAL);
  twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  esp_err_t err = twai_driver_install(&g_config, &t_config, &f_config);
  if (err != ESP_OK) {
    Serial.printf("Fehler beim CAN-Setup: %d\n", err);
    while (true) delay(1000);
  }

  err = twai_start();
  if (err != ESP_OK) {
    Serial.printf("Fehler beim CAN-Start: %d\n", err);
    while (true) delay(1000);
  }

  Serial.printf("CAN OK: TX=%d, RX=%d, 500 kbit/s\n", CAN_TX_GPIO, CAN_RX_GPIO);
}

uint8_t classToCanCmd(int class_index) {
  switch (class_index) {
    case 0: return CAN_CMD_LICHT_AN;
    case 1: return CAN_CMD_LICHT_AUS;
    default: return CAN_CMD_UNKNOWN;
  }
}

bool sendCanCommand(uint8_t cmd, uint8_t confidence_pct) {
  twai_message_t message = {};
  message.identifier = CAN_ID_VOICE;
  message.extd = 0;
  message.rtr = 0;
  message.data_length_code = 2;
  message.data[0] = cmd;
  message.data[1] = confidence_pct;

  esp_err_t err = twai_transmit(&message, pdMS_TO_TICKS(200));
  if (err != ESP_OK) {
    Serial.printf("Fehler beim CAN-Send: %d\n", err);
    return false;
  }

  Serial.printf("CAN gesendet: ID=0x%03X, cmd=0x%02X, confidence=%u%%\n",
                  CAN_ID_VOICE, cmd, confidence_pct);
  return true;
}

void handleResult(const ClassificationResult *result) {
  Serial.printf("Ergebnis: %s (%.1f%%)\n",
                result->predicted_label, result->confidence * 100.0f);

  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.printf("  P(%s) = %.3f\n", CLASS_NAMES[i], result->probabilities[i]);
  }

  if (result->signal_rms < MIN_SIGNAL_RMS) {
    Serial.printf("WARNUNG: Signal zu leise (RMS=%.4f)\n", result->signal_rms);
    return;
  }

  if (result->class_index == 2 || result->confidence < MIN_CONFIDENCE) {
    Serial.printf("Kein gueltiger Befehl (unknown oder confidence < %.0f%%)\n",
                    MIN_CONFIDENCE * 100.0f);
    return;
  }

  uint8_t cmd = classToCanCmd(result->class_index);
  uint8_t confidence_pct = (uint8_t)(result->confidence * 100.0f);
  sendCanCommand(cmd, confidence_pct);
}

bool buttonPressed() {
  return gpio_get_level((gpio_num_t)BUTTON_GPIO) == 0;
}

static void setupButtonInput() {
  gpio_reset_pin((gpio_num_t)BUTTON_GPIO);
  gpio_config_t button_cfg = {
    .pin_bit_mask = 1ULL << BUTTON_GPIO,
    .mode = GPIO_MODE_INPUT,
    .pull_up_en = GPIO_PULLUP_DISABLE,
    .pull_down_en = GPIO_PULLDOWN_DISABLE,
    .intr_type = GPIO_INTR_DISABLE
  };
  ESP_ERROR_CHECK(gpio_config(&button_cfg));
}

void waitForButtonPress() {
  stopI2SDriver();
  setupButtonInput();

  Serial.printf("Warte auf Taster GPIO %d (Pegel=%d, 0=gedrueckt)...\n",
                BUTTON_GPIO, gpio_get_level((gpio_num_t)BUTTON_GPIO));

  uint32_t hang_ms = 0;
  while (buttonPressed()) {
    hang_ms += 10;
    if (hang_ms == 3000) {
      Serial.println("Hinweis: Taster haengt auf LOW – externen 10k Pull-up an 3V3 pruefen.");
    }
    delay(10);
  }

  while (true) {
    if (buttonPressed()) {
      delay(BUTTON_DEBOUNCE_MS);
      if (buttonPressed()) {
        Serial.println("Taster erkannt.");
        while (buttonPressed()) {
          delay(10);
        }
        return;
      }
    }
    delay(20);
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("Sprachsteuerung Start");

  setupButtonInput();

  audio_buffer = (int16_t *)malloc(SAMPLES_COUNT * sizeof(int16_t));
  model_input_buffer = (float *)malloc(N_MELS * REFERENCE_WIDTH * sizeof(float));
  if (!audio_buffer || !model_input_buffer) {
    Serial.printf("Fehler: Audio-Buffer Allokierung (audio=%u B, frei: %u Bytes)\n",
                  (unsigned)(SAMPLES_COUNT * sizeof(int16_t)),
                  (unsigned)esp_get_free_heap_size());
    while (true) delay(1000);
  }

  esp_err_t err = dsps_fft2r_init_fc32(NULL, 1024);
  if (err != ESP_OK) {
    Serial.printf("Fehler bei FFT-Init: %d\n", err);
    while (true) delay(1000);
  }

  setupCan();
  setupModel();
  setupI2S();

  Serial.printf("Audio: %.1fs int16 (%u B), freier Heap: %u Bytes\n",
                DURATION, (unsigned)(SAMPLES_COUNT * sizeof(int16_t)),
                (unsigned)esp_get_free_heap_size());

  Serial.printf("Bereit. Taster (GPIO %d) druecken, dann sprechen.\n", BUTTON_GPIO);
}

void loop() {
  waitForButtonPress();

  recordMicrophone(audio_buffer, SAMPLES_COUNT);

  ClassificationResult result = {};
  classifyAudio(audio_buffer, SAMPLES_COUNT, &result);
  handleResult(&result);

  Serial.println("----------------------------------------");
}

extern "C" void app_main(void) {
  xTaskCreatePinnedToCore(
      [](void *) {
        setup();
        while (true) {
          loop();
        }
      },
      "voice_main",
      32768,
      nullptr,
      1,
      nullptr,
      0);
}
