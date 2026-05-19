from micropython import const


# Button / GPIO
BUTTON_PIN = const(39) 
BUTTON_ACTIVE_LOW = const(1)
DEBOUNCE_MS = const(180)

# I2S microphone pins (example wiring for SPH0645 / similar)
I2S_ID = const(0)

I2S_SCK_PIN = const(17)   # BCLK
I2S_WS_PIN  = const(18)   # LRCL
I2S_SD_PIN  = const(21)   # DOUT

I2S_BUFFER_BYTES = const(4096)
I2S_BITS = const(32)

# Audio
SAMPLE_RATE = const(16000)
RECORD_SECONDS = const(1)
NUM_SAMPLES = const(SAMPLE_RATE * RECORD_SECONDS)
CHUNK_SAMPLES = const(256)
PRE_EMPHASIS = 0.97

# MFCC
FRAME_MS = const(25)
HOP_MS = const(10)
FRAME_LEN = const((SAMPLE_RATE * FRAME_MS) // 1000)   # 400
HOP_LEN = const((SAMPLE_RATE * HOP_MS) // 1000)       # 160
N_FFT = const(512)
NUM_MELS = const(20)
NUM_MFCC = const(13)
MFCC_MIN_HZ = 20.0
MFCC_MAX_HZ = 8000.0
FRAME_COUNT = const(((NUM_SAMPLES - FRAME_LEN) // HOP_LEN) + 1)

# Model
MODEL_PATH = "model.dat"

# Output pins
PIN_BLINKER_LINKS = const(14)
PIN_BLINKER_RECHTS = const(27)
PIN_INNENLICHT = const(12)
PIN_LICHT = const(13)

# Class labels
CLASS_LABELS = {
    0: "blinker_links",
    1: "blinker_rechts",
    2: "innenbeleuchtung_an",
    3: "innenbeleuchtung_aus",
    4: "licht_an",
    5: "licht_aus",
}

