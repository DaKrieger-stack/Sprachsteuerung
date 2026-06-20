/**
 * @file main.cpp
 * @brief ESP32-Sprachsteuerung: I2S-Aufnahme, Mel-Spektrogramm, TFLite-Inferenz, CAN-Ausgabe.
 *
 * Pipeline: SPH0645 (I2S) → Audio-Vorverarbeitung → Mel-Spektrogramm → TFLite-CNN → TWAI/CAN.
 *
 * Die Mel-Parameter müssen mit `scripts/export_esp32_model.py` übereinstimmen.
 *
 * @see modell.h
 */

#include <driver/gpio.h>#include <driver/i2s.h>
#include <driver/twai.h>
#include <esp_dsp.h>
#include <esp_log.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modell.h"

static const char *TAG = "voice";

/**
 * @defgroup hw Hardware- und Pipeline-Konstanten
 * @brief Pinbelegung, Audio- und Modell-Parameter.
 *
 * Werte müssen mit `export_esp32_model.py` synchron gehalten werden.
 * @{ */

#define I2S_PORT         I2S_NUM_0
#define I2S_DOUT         21   /**< I2S Daten vom SPH0645 */
#define I2S_BCLK         17   /**< I2S Bit-Takt */
#define I2S_LRCL         18   /**< I2S Word-Select / LRCK */
#define CAN_TX_GPIO      32   /**< TWAI TX */
#define CAN_RX_GPIO      25   /**< TWAI RX */
#define BUTTON_GPIO      39   /**< Taster, aktiv low */

#define SAMPLE_RATE      16000 /**< Abtastrate in Hz */
#define BUFFER_LEN       512   /**< I2S DMA-Pufferlänge (Samples) */
#define DURATION_S       2.0f  /**< Aufnahmedauer in Sekunden */
#define SAMPLES_COUNT    ((size_t)(DURATION_S * SAMPLE_RATE)) /**< 32000 @ 16 kHz */

#define N_MELS           64    /**< Mel-Bänder */
#define N_FFT            512   /**< FFT-Fensterlänge */
#define HOP_LENGTH       256   /**< Hop zwischen Frames */
#define SPEC_WIDTH       124   /**< Feste Zeitachse fürs Modell */
#define NUM_CLASSES      2     /**< licht_an, licht_aus */
#define CLASS_UNKNOWN    2     /**< Kein sicherer Treffer */

#define MIN_SIGNAL_RMS   0.01f /**< Unterhalb: Aufnahme ignorieren */
#define MIN_CONF_AN      0.45f /**< Mindest-Konfidenz licht_an */
#define MIN_CONF_AUS     0.30f /**< Mindest-Konfidenz licht_aus (ponytail: INT8+Live-Mikro) */
#define MIN_MARGIN       0.05f /**< Mindest-Abstand zwischen Top-2-Klassen */
#define CAN_ID_VOICE     0x100  /**< CAN-Identifier Sprachbefehle */
#define DEBOUNCE_MS      50    /**< Taster-Entprellzeit */

static_assert(SAMPLES_COUNT == 32000, "2s @ 16 kHz");

#define ARENA_BYTES      (128 * 1024) /**< TFLite Micro Arena-Größe */

#define LOG_DIV  "========================================"

static const char *CLASS_NAMES[] = {"licht_an", "licht_aus", "unknown"};

/** @} */

/**
 * @defgroup audio Audio-Verarbeitung
 * @brief Konvertierung, Vorverarbeitung und Mel-Spektrogramm.
 * @{ */

// ponytail: audio+spec on heap; arena static (164 KB malloc blew ~171 KB heap)
static int16_t *g_audio;
static float *g_spec;
static float *g_fft;
static float *g_power;
static uint8_t g_arena[ARENA_BYTES];
static float g_hann[N_FFT];

static tflite::MicroInterpreter *g_interpreter;
static TfLiteTensor *g_input;

static bool g_i2s_up;

// --- tiny helpers ---
static inline float s16_to_f(int16_t s) { return (float)s / 32768.0f; }

static inline int16_t i2s_to_s16(int32_t raw) {
  int32_t v = raw >> 16;
  if (v > 32767) return 32767;
  if (v < -32768) return -32768;
  return (int16_t)v;
}

static inline int32_t i2s_pick(int32_t l, int32_t r) {
  int32_t al = l < 0 ? -l : l, ar = r < 0 ? -r : r;
  return al >= ar ? l : r;
}

/**
 * @brief Berechnet den RMS-Wert eines int16-Audiobuffers (normalisiert auf [-1, 1]).
 * @param buf PCM-Samples
 * @param n   Anzahl Samples
 * @return RMS-Amplitude (0 bei leerem Buffer)
 */
static float audio_rms(const int16_t *buf, size_t n) {  if (!n) return 0.0f;
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    float s = s16_to_f(buf[i]);
    sum += (double)s * s;
  }
  return (float)sqrt(sum / (double)n);
}

/**
 * @brief Entfernt führende und nachfolgende Stille in-place.
 * @param buf        PCM-Buffer (wird verschoben)
 * @param len        Länge in Samples
 * @param thresh_db  Schwellwert in dBFS (z. B. -40)
 * @return Neue Länge nach dem Trimmen
 * @note ponytail: O(n)-Scan; Upgrade: librosa-style Split falls nötig
 */
static size_t trim_silence(int16_t *buf, size_t len, float thresh_db) {  float t = powf(10.0f, thresh_db / 20.0f);
  size_t start = 0, end = len;
  for (size_t i = 0; i < len; i++) {
    if (fabsf(s16_to_f(buf[i])) > t) { start = i; break; }
  }
  for (size_t i = len; i > 0; i--) {
    if (fabsf(s16_to_f(buf[i - 1])) > t) { end = i; break; }
  }
  if (end <= start) return len;
  size_t n = end - start;
  memmove(buf, buf + start, n * sizeof(int16_t));
  return n;
}

/**
 * @brief Vorverarbeitung: Pre-Emphasis (0.97), Peak-Normalisierung auf int16.
 * @param buf PCM-Buffer in-place
 * @param len Anzahl Samples
 */
static void prep_audio(int16_t *buf, size_t len) {  const float pre = 0.97f;
  for (size_t i = len - 1; i > 0; i--) {
    float v = s16_to_f(buf[i]) - pre * s16_to_f(buf[i - 1]);
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    buf[i] = (int16_t)lroundf(v * 32767.0f);
    if ((i & 4095) == 0) vTaskDelay(1);
  }
  int16_t peak = 0;
  for (size_t i = 0; i < len; i++) {
    int16_t a = buf[i] < 0 ? (int16_t)-buf[i] : buf[i];
    if (a > peak) peak = a;
  }
  if (peak) {
    for (size_t i = 0; i < len; i++) {
      buf[i] = (int16_t)((int32_t)buf[i] * 32767 / peak);
    }
  }
}

static inline float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static float mel_weight(int m, int k, float mel_min, float mel_step) {
  float f = (float)k * (float)SAMPLE_RATE / (float)N_FFT;
  float left = mel_to_hz(mel_min + m * mel_step);
  float mid = mel_to_hz(mel_min + (m + 1) * mel_step);
  float right = mel_to_hz(mel_min + (m + 2) * mel_step);
  if (f >= left && f <= mid) return (f - left) / (mid - left);
  if (f >= mid && f <= right) return (right - f) / (right - mid);
  return 0.0f;
}

/**
 * @brief Erzeugt ein log-Mel-Spektrogramm mit Hann-Fenster und Min-Max-Normalisierung.
 * @param audio      Eingabe-PCM
 * @param audio_len  Länge in Samples
 * @param out        Ausgabe-Buffer, Layout `[mel * frames + frame]` (row-major)
 * @param max_frames Maximale Zeitachsen-Frames (typisch @ref SPEC_WIDTH)
 * @return Tatsächliche Anzahl Frames
 */
static int mel_spec(const int16_t *audio, size_t audio_len, float *out, int max_frames) {  static bool hann_ok;
  if (!hann_ok) {
    dsps_wind_hann_f32(g_hann, N_FFT);
    hann_ok = true;
  }

  int frames = (int)((audio_len - N_FFT) / HOP_LENGTH) + 1;
  if (frames <= 0) frames = 1;
  if (frames > max_frames) frames = max_frames;

  float lo = hz_to_mel(0.0f), hi = hz_to_mel(SAMPLE_RATE / 2.0f);
  float step = (hi - lo) / (N_MELS + 1);
  float gmax = -INFINITY, gmin = INFINITY;

  for (int fr = 0; fr < frames; fr++) {
    int start = fr * HOP_LENGTH;
    for (int i = 0; i < N_FFT; i++) {
      g_fft[i * 2] = (start + i < (int)audio_len)
                         ? s16_to_f(audio[start + i]) * g_hann[i]
                         : 0.0f;
      g_fft[i * 2 + 1] = 0.0f;
    }
    dsps_fft2r_fc32(g_fft, N_FFT);
    dsps_bit_rev_fc32(g_fft, N_FFT);
    dsps_cplx2reC_fc32(g_fft, N_FFT);

    for (int i = 0; i <= N_FFT / 2; i++) {
      float re = g_fft[i * 2], im = g_fft[i * 2 + 1];
      g_power[i] = re * re + im * im;
    }

    for (int m = 0; m < N_MELS; m++) {
      float e = 0.0f;
      for (int k = 0; k <= N_FFT / 2; k++) {
        float w = mel_weight(m, k, lo, step);
        if (w > 0.0f) e += w * g_power[k];
      }
      if (e < 1e-10f) e = 1e-10f;
      float lm = 10.0f * log10f(e);
      out[m * frames + fr] = lm;
      if (lm > gmax) gmax = lm;
      if (lm < gmin) gmin = lm;
    }
    vTaskDelay(1);
  }

  float range = gmax - gmin;
  if (range < 1e-8f) range = 1e-8f;
  for (int m = 0; m < N_MELS; m++) {
    for (int t = 0; t < frames; t++) {
      out[m * frames + t] = (out[m * frames + t] - gmin) / range;
    }
  }
  return frames;
}

/**
 * @brief Passt die Zeitachse des Spektrogramms auf @ref SPEC_WIDTH an (zentriertes Croppen oder Zero-Pad).
 * @param spec  Mel-Spektrogramm in-place
 * @param cur_w Aktuelle Frame-Anzahl
 */
static void fit_spec_width(float *spec, int cur_w) {  if (cur_w == SPEC_WIDTH) return;
  if (cur_w > SPEC_WIDTH) {
    int off = (cur_w - SPEC_WIDTH) / 2;
    for (int m = 0; m < N_MELS; m++) {
      memmove(spec + m * SPEC_WIDTH, spec + m * cur_w + off, SPEC_WIDTH * sizeof(float));
    }
    return;
  }
  for (int m = N_MELS - 1; m >= 0; m--) {
    memmove(spec + m * SPEC_WIDTH, spec + m * cur_w, cur_w * sizeof(float));
    memset(spec + m * SPEC_WIDTH + cur_w, 0, (SPEC_WIDTH - cur_w) * sizeof(float));
  }
}

/** @} */

/**
 * @defgroup i2s I2S-Aufnahme
 * @brief SPH0645-Mikrofon über ESP-IDF I2S-Treiber.
 * @{ */

/**
 * @brief Installiert und startet den I2S-RX-Treiber (32-bit, Stereo-Pick).
 * @return ESP_OK bei Erfolg
 */
static esp_err_t i2s_start(void) {  i2s_config_t cfg = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = BUFFER_LEN,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_pin_config_t pins = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = I2S_BCLK,
      .ws_io_num = I2S_LRCL,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_DOUT};

  esp_err_t err = i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  if (err != ESP_OK) return err;
  err = i2s_set_pin(I2S_PORT, &pins);
  if (err != ESP_OK) {
    i2s_driver_uninstall(I2S_PORT);
    return err;
  }
  i2s_zero_dma_buffer(I2S_PORT);
  vTaskDelay(pdMS_TO_TICKS(100));
  g_i2s_up = true;
  return ESP_OK;
}

static void i2s_stop(void) {
  if (!g_i2s_up) return;
  i2s_driver_uninstall(I2S_PORT);
  g_i2s_up = false;
}

static void i2s_warmup(void) {
  int32_t trash[BUFFER_LEN];
  size_t n;
  for (int i = 0; i < 6; i++) {
    i2s_read(I2S_PORT, trash, sizeof(trash), &n, pdMS_TO_TICKS(50));
  }
}

static float i2s_rms_norm(TickType_t timeout) {
  int32_t raw[BUFFER_LEN];
  size_t bytes = 0;
  i2s_read(I2S_PORT, raw, sizeof(raw), &bytes, timeout);
  int n = (int)(bytes / sizeof(int32_t));
  if (n <= 0) return 0.0f;

  double sum = 0.0;
  int count = 0;
  if (n >= 2) {
    for (int i = 0; i + 1 < n; i += 2) {
      int32_t s = i2s_pick(raw[i], raw[i + 1]) >> 8;
      sum += (double)s * s;
      count++;
    }
  } else {
    for (int i = 0; i < n; i++) {
      int32_t s = raw[i] >> 8;
      sum += (double)s * s;
      count++;
    }
  }
  return count ? (float)(sqrt(sum / count) / 8388607.0) : 0.0f;
}

/**
 * @brief Nimmt exakt @p need Samples bei @ref SAMPLE_RATE auf.
 * @param out  Zielpuffer (int16 PCM)
 * @param need Anzahl Samples (typisch @ref SAMPLES_COUNT)
 */
static void record(int16_t *out, size_t need) {  if (!g_i2s_up && i2s_start() != ESP_OK) {
    ESP_LOGE(TAG, "I2S start failed");
    return;
  }
  i2s_warmup();

  ESP_LOGD(TAG, "Aufnahme %.1fs start (pre-RMS=%.4f)", DURATION_S,
           i2s_rms_norm(pdMS_TO_TICKS(200)));

  size_t got = 0;
  int32_t raw[BUFFER_LEN];
  int32_t peak = 0;

  while (got < need) {
    size_t bytes = 0;
    i2s_read(I2S_PORT, raw, sizeof(raw), &bytes, portMAX_DELAY);
    int n = (int)(bytes / sizeof(int32_t));

    if (n >= 2) {
      for (int i = 0; i + 1 < n && got < need; i += 2) {
        int32_t pick = i2s_pick(raw[i], raw[i + 1]);
        int32_t a = pick < 0 ? -pick : pick;
        if (a > peak) peak = a;
        out[got++] = i2s_to_s16(pick);
      }
    } else {
      for (int i = 0; i < n && got < need; i++) {
        int32_t a = raw[i] < 0 ? -raw[i] : raw[i];
        if (a > peak) peak = a;
        out[got++] = i2s_to_s16(raw[i]);
      }
    }
  }
  ESP_LOGD(TAG, "Aufnahme fertig RMS=%.4f peak=%ld", audio_rms(out, need), (long)peak);
}

/** @} */

/**
 * @defgroup ml TensorFlow Lite
 * @brief Modell-Laden, Quantisierung und Klassifikation.
 * @{ */

/**
 * @brief Lädt @ref tiny_cnn_model_big_tflite, registriert Ops und alloziert Tensoren.
 * Blockiert bei Schema- oder Speicherfehler.
 */
static void model_init(void) {  const tflite::Model *model = tflite::GetModel(tiny_cnn_model_big_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Modell-Schema passt nicht");
    for (;;) vTaskDelay(pdMS_TO_TICKS(5000));
  }

  static tflite::MicroMutableOpResolver<12> ops;
  ops.AddConv2D();
  ops.AddMaxPool2D();
  ops.AddMean();
  ops.AddAdd();
  ops.AddMul();
  ops.AddFullyConnected();
  ops.AddSoftmax();
  ops.AddQuantize();
  ops.AddDequantize();

  static tflite::MicroInterpreter interp(model, ops, g_arena, ARENA_BYTES);
  g_interpreter = &interp;
  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "tensor alloc failed — Arena %u B, genutzt %u B",
             (unsigned)ARENA_BYTES, g_interpreter->arena_used_bytes());
    for (;;) vTaskDelay(pdMS_TO_TICKS(5000));
  }
  g_input = g_interpreter->input(0);
  TfLiteTensor *out_t = g_interpreter->output(0);
  ESP_LOGI(TAG, "Modell OK (%u B, Arena %u/%u B)", tiny_cnn_model_big_tflite_len,
           g_interpreter->arena_used_bytes(), (unsigned)ARENA_BYTES);
  ESP_LOGI(TAG, "Tensor in=%d out=%d out_dims=%d", (int)g_input->type,
           (int)out_t->type, out_t->dims ? out_t->dims->data[1] : -1);
}

static void write_input(TfLiteTensor *t, const float *spec, size_t n) {
  if (t->type == kTfLiteFloat32) {
    memcpy(t->data.f, spec, n * sizeof(float));
    return;
  }
  float scale = t->params.scale;
  int zp = t->params.zero_point;
  if (t->type == kTfLiteUInt8) {
    for (size_t i = 0; i < n; i++) {
      int v = (int)lroundf(spec[i] / scale) + zp;
      if (v < 0) v = 0;
      if (v > 255) v = 255;
      t->data.uint8[i] = (uint8_t)v;
    }
    return;
  }
  if (t->type == kTfLiteInt8) {
    for (size_t i = 0; i < n; i++) {
      int v = (int)lroundf(spec[i] / scale) + zp;
      if (v < -128) v = -128;
      if (v > 127) v = 127;
      t->data.int8[i] = (int8_t)v;
    }
  }
}

static void read_probs(TfLiteTensor *t, float *probs, int n) {
  if (t->type == kTfLiteFloat32) {
    for (int i = 0; i < n; i++) probs[i] = t->data.f[i];
    return;
  }
  float scale = t->params.scale;
  int zp = t->params.zero_point;
  if (t->type == kTfLiteUInt8) {
    for (int i = 0; i < n; i++)
      probs[i] = scale * ((int)t->data.uint8[i] - zp);
  } else if (t->type == kTfLiteInt8) {
    for (int i = 0; i < n; i++)
      probs[i] = scale * ((int)t->data.int8[i] - zp);
  }
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += probs[i];
  if (sum > 1e-6f)
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

/**
 * @brief Wählt die Klasse anhand Konfidenz-Schwellen und Mindest-Abstand.
 * @param probs Normalisierte Klassenwahrscheinlichkeiten [licht_an, licht_aus]
 * @param conf  Ausgabe: Konfidenz der gewählten Klasse
 * @return Klassenindex (0/1) oder @ref CLASS_UNKNOWN
 */
static int decide_class(const float *probs, float *conf) {  float p_an = probs[0], p_aus = probs[1];
  if (p_aus > p_an) {
    *conf = p_aus;
    if (p_aus >= MIN_CONF_AUS && (p_aus - p_an) >= MIN_MARGIN) return 1;
  } else {
    *conf = p_an;
    if (p_an >= MIN_CONF_AN && (p_an - p_aus) >= MIN_MARGIN) return 0;
  }
  return CLASS_UNKNOWN;
}

/**
 * @brief Vollständige Inferenz-Pipeline für einen Audioblock.
 * @param audio    PCM (wird vorverarbeitet)
 * @param len      Länge in Samples
 * @param best_idx Ausgabe: Klassenindex oder @ref CLASS_UNKNOWN
 * @param best_p   Ausgabe: Konfidenz
 * @param rms_out  Ausgabe: RMS vor der Vorverarbeitung
 * @param probs    Ausgabe: Klassenwahrscheinlichkeiten (@ref NUM_CLASSES Einträge)
 * @return false bei Spektrogramm- oder Invoke-Fehler
 */
static bool classify(int16_t *audio, size_t len, int *best_idx, float *best_p,
                     float *rms_out, float *probs) {  *rms_out = audio_rms(audio, len);
  prep_audio(audio, len);
  size_t trimmed = trim_silence(audio, len, -40.0f);

  int frames = mel_spec(audio, trimmed, g_spec, SPEC_WIDTH);
  if (frames <= 0) return false;
  if (frames != SPEC_WIDTH) fit_spec_width(g_spec, frames);

  write_input(g_input, g_spec, N_MELS * SPEC_WIDTH);
  if (g_interpreter->Invoke() != kTfLiteOk) return false;

  read_probs(g_interpreter->output(0), probs, NUM_CLASSES);
  *best_idx = decide_class(probs, best_p);
  return true;
}

/** @} */

/**
 * @defgroup can CAN-Bus (TWAI)
 * @brief Befehlsübertragung an den Licht-Controller.
 * @{ */

/**
 * @brief Initialisiert TWAI mit 500 kbit/s und Accept-All-Filter.
 */
static void can_init(void) {  twai_general_config_t g =
      TWAI_GENERAL_CONFIG_DEFAULT((gpio_num_t)CAN_TX_GPIO, (gpio_num_t)CAN_RX_GPIO,
                                  TWAI_MODE_NORMAL);
  twai_timing_config_t t = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t f = TWAI_FILTER_CONFIG_ACCEPT_ALL();
  ESP_ERROR_CHECK(twai_driver_install(&g, &t, &f));
  ESP_ERROR_CHECK(twai_start());
  ESP_LOGD(TAG, "CAN TX=%d RX=%d 500k", CAN_TX_GPIO, CAN_RX_GPIO);
}

/**
 * @brief Sendet einen Sprachbefehl per CAN.
 * @param cmd Befehl: `0x01` licht_an, `0x02` licht_aus
 * @param pct Konfidenz 0–100 (Byte 1)
 * @return true bei erfolgreicher Übertragung
 */
static bool can_send(uint8_t cmd, uint8_t pct) {  twai_message_t msg = {};
  msg.identifier = CAN_ID_VOICE;
  msg.data_length_code = 2;
  msg.data[0] = cmd;
  msg.data[1] = pct;
  esp_err_t err = twai_transmit(&msg, pdMS_TO_TICKS(200));
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "CAN tx %d", err);
    return false;
  }
  ESP_LOGD(TAG, "CAN gesendet cmd=0x%02X conf=%u%%", cmd, pct);
  return true;
}

/** @} */

/**
 * @defgroup ui Taster-Eingabe
 * @brief GPIO 39, aktiv low, mit Entprellung.
 * @{ */

/**
 * @brief Konfiguriert den Taster-Pin als Eingang ohne Pull.
 */
static void button_init(void) {  gpio_reset_pin((gpio_num_t)BUTTON_GPIO);
  gpio_config_t c = {
      .pin_bit_mask = 1ULL << BUTTON_GPIO,
      .mode = GPIO_MODE_INPUT,
      .pull_up_en = GPIO_PULLUP_DISABLE,
      .pull_down_en = GPIO_PULLDOWN_DISABLE,
      .intr_type = GPIO_INTR_DISABLE};
  ESP_ERROR_CHECK(gpio_config(&c));
}

static bool button_down(void) {
  return gpio_get_level((gpio_num_t)BUTTON_GPIO) == 0;
}

/**
 * @brief Blockiert bis der Taster einmal gedrückt und wieder losgelassen wurde.
 * Stoppt I2S während des Wartens (Stromsparen).
 */
static void wait_button(void) {  i2s_stop();
  button_init();
  ESP_LOGI(TAG, LOG_DIV);
  ESP_LOGI(TAG, "BEREIT — Taste GPIO%d druecken", BUTTON_GPIO);
  ESP_LOGI(TAG, LOG_DIV);

  while (button_down()) vTaskDelay(pdMS_TO_TICKS(10));

  for (;;) {
    if (button_down()) {
      vTaskDelay(pdMS_TO_TICKS(DEBOUNCE_MS));
      if (button_down()) {
        while (button_down()) vTaskDelay(pdMS_TO_TICKS(10));
        return;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

/** @} */

/**
 * @defgroup app Anwendung
 * @brief Hauptschleife, Ergebnis-Auswertung und ESP-IDF-Einstieg.
 * @{ */

/**
 * @brief Loggt das Klassifikationsergebnis und sendet ggf. CAN.
 *
 * Ignoriert zu leise Aufnahmen (@ref MIN_SIGNAL_RMS) und @ref CLASS_UNKNOWN.
 */
static void handle(int class_idx, float conf, float rms, const float *probs) {  ESP_LOGI(TAG, LOG_DIV);
  ESP_LOGI(TAG, "ERGEBNIS: %s (%.0f%%)", CLASS_NAMES[class_idx], conf * 100.0f);
  ESP_LOGI(TAG, "  licht_an=%.0f%%  licht_aus=%.0f%%", probs[0] * 100.0f, probs[1] * 100.0f);
  ESP_LOGI(TAG, "  RMS=%.4f", rms);

  if (rms < MIN_SIGNAL_RMS) {
    ESP_LOGW(TAG, ">>> IGNORIERT: zu leise <<<");
    ESP_LOGI(TAG, LOG_DIV);
    return;
  }
  if (class_idx == CLASS_UNKNOWN) {
    ESP_LOGW(TAG, ">>> IGNORIERT: unknown (an=%.0f%% aus=%.0f%%, braucht an>=%.0f%% aus>=%.0f%% + %.0f%% Abstand) <<<",
             probs[0] * 100.0f, probs[1] * 100.0f,
             MIN_CONF_AN * 100.0f, MIN_CONF_AUS * 100.0f, MIN_MARGIN * 100.0f);
    ESP_LOGI(TAG, LOG_DIV);
    return;
  }
  static const uint8_t cmds[] = {0x01, 0x02, 0x00};
  if (can_send(cmds[class_idx], (uint8_t)(conf * 100.0f))) {
    ESP_LOGI(TAG, ">>> CAN GESENDET: %s <<<", CLASS_NAMES[class_idx]);
  } else {
    ESP_LOGW(TAG, ">>> CAN FEHLER <<<");
  }
  ESP_LOGI(TAG, LOG_DIV);
}

/**
 * @brief FreeRTOS-Haupttask: init → Endlosschleife (Taste → Aufnahme → Klassifikation).
 * @param unused Task-Parameter (ungenutzt)
 */
static void voice_task(void *) {  ESP_LOGI(TAG, "freier Heap: %u B", (unsigned)esp_get_free_heap_size());

  g_audio = (int16_t *)malloc(SAMPLES_COUNT * sizeof(int16_t));
  g_spec = (float *)malloc(N_MELS * SPEC_WIDTH * sizeof(float));
  g_fft = (float *)malloc(2 * N_FFT * sizeof(float));
  g_power = (float *)malloc((N_FFT / 2 + 1) * sizeof(float));
  if (!g_audio) ESP_LOGE(TAG, "malloc audio failed (braucht %u B)",
                          (unsigned)(SAMPLES_COUNT * sizeof(int16_t)));
  if (!g_spec) ESP_LOGE(TAG, "malloc spec failed");
  if (!g_fft) ESP_LOGE(TAG, "malloc fft failed");
  if (!g_power) ESP_LOGE(TAG, "malloc power failed");
  if (!g_audio || !g_spec || !g_fft || !g_power) {
    ESP_LOGE(TAG, "Speicher voll — Heap noch %u B frei",
             (unsigned)esp_get_free_heap_size());
    for (;;) vTaskDelay(pdMS_TO_TICKS(5000));
  }

  ESP_ERROR_CHECK(dsps_fft2r_init_fc32(NULL, 1024));
  button_init();
  can_init();
  model_init();
  if (i2s_start() != ESP_OK) {
    ESP_LOGE(TAG, "I2S start fehlgeschlagen");
    for (;;) vTaskDelay(pdMS_TO_TICKS(5000));
  }
  ESP_LOGI(TAG, LOG_DIV);
  ESP_LOGI(TAG, "SYSTEM START — Sprachsteuerung aktiv");
  ESP_LOGI(TAG, "Ablauf: Taste -> %.0fs sprechen -> Ergebnis", DURATION_S);
  ESP_LOGI(TAG, LOG_DIV);

  for (;;) {
    wait_button();
    ESP_LOGI(TAG, LOG_DIV);
    ESP_LOGI(TAG, ">>> JETZT SPRECHEN! (%.0f Sekunden) <<<", DURATION_S);
    ESP_LOGI(TAG, LOG_DIV);
    record(g_audio, SAMPLES_COUNT);
    ESP_LOGI(TAG, "Aufnahme fertig — klassifiziere...");

    int idx = CLASS_UNKNOWN;
    float conf = 0.0f, rms = 0.0f, probs[NUM_CLASSES] = {0};
    if (classify(g_audio, SAMPLES_COUNT, &idx, &conf, &rms, probs)) {
      handle(idx, conf, rms, probs);
    } else {
      ESP_LOGE(TAG, LOG_DIV);
      ESP_LOGE(TAG, ">>> FEHLER: Klassifikation fehlgeschlagen <<<");
      ESP_LOGE(TAG, LOG_DIV);
    }
  }
}

/**
 * @brief ESP-IDF-Einstiegspunkt — startet @ref voice_task auf Core 0.
 */
extern "C" void app_main(void) {
  // ponytail: 32768 Worte = 128 KB Stack fraß den Heap; 8192 = 32 KB reicht
  xTaskCreatePinnedToCore(voice_task, "voice", 8192, NULL, 1, NULL, 0);
}

/** @} */