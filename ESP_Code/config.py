from micropython import const


# I2S microphone wiring
I2S_ID = const(0)
I2S_SCK_PIN = const(17)
I2S_WS_PIN = const(18)
I2S_SD_PIN = const(21)

# Many MEMS microphones output 24-bit samples in a 32-bit I2S frame.
# Keep this configurable because some MicroPython builds only support 32-bit RX.
I2S_BITS = const(32)
I2S_BUFFER_BYTES = const(8192)

# Audio capture
SAMPLE_RATE = const(16000)
RECORD_SECONDS = const(1)
NUM_SAMPLES = const(SAMPLE_RATE * RECORD_SECONDS)
CHUNK_SAMPLES = const(256)

# Optional DC removal / pre emphasis
PRE_EMPHASIS = 0.97

# Feature extraction
FRAME_MS = const(25)
HOP_MS = const(20)
FRAME_LEN = const((SAMPLE_RATE * FRAME_MS) // 1000)   # 400
HOP_LEN = const((SAMPLE_RATE * HOP_MS) // 1000)       # 320
N_FFT = const(512)
NUM_MELS = const(20)
FRAME_COUNT = const(((NUM_SAMPLES - FRAME_LEN) // HOP_LEN) + 1)  # 49
MEL_MIN_HZ = 50.0
MEL_MAX_HZ = 7600.0

# Decision logic
MIN_SIGNAL_RMS = 900.0
SIMILARITY_THRESHOLD = 0.78
MARGIN_THRESHOLD = 0.04

# Template storage
TEMPLATE_STORE_PATH = "command_templates.json"
MAX_TEMPLATES_PER_CLASS = const(6)
COMMAND_ON = "ON"
COMMAND_OFF = "OFF"
COMMANDS = (COMMAND_ON, COMMAND_OFF)

# CAN output
CAN_ID = const(0)
CAN_TX_PIN = const(32)
CAN_RX_PIN = const(26)
CAN_BAUDRATE = const(500000)
CAN_ARB_ID_LIGHT = const(0x123)

# Payload format:
# byte 0 = light state
# 0x01 -> light on
# 0x00 -> light off
CAN_PAYLOAD_LIGHT_ON = b"\x01"
CAN_PAYLOAD_LIGHT_OFF = b"\x00"
