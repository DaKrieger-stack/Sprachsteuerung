from array import array
from machine import I2S, Pin

import config


class AudioCapture:
    def __init__(self):
        self._raw_chunk = bytearray(config.CHUNK_SAMPLES * 4)
        self._i2s = I2S(
            config.I2S_ID,
            sck=Pin(config.I2S_SCK_PIN),
            ws=Pin(config.I2S_WS_PIN),
            sd=Pin(config.I2S_SD_PIN),
            mode=I2S.RX,
            bits=config.I2S_BITS,
            format=I2S.MONO,
            rate=config.SAMPLE_RATE,
            ibuf=config.I2S_BUFFER_BYTES,
        )

    def deinit(self):
        self._i2s.deinit()

    def capture_into(self, target_samples):
        sample_index = 0
        total_samples = len(target_samples)
        raw = self._raw_chunk

        while sample_index < total_samples:
            bytes_read = self._i2s.readinto(raw)
            if not bytes_read:
                continue

            limit = bytes_read // 4
            for i in range(limit):
                base = i * 4
                word = (
                    raw[base]
                    | (raw[base + 1] << 8)
                    | (raw[base + 2] << 16)
                    | (raw[base + 3] << 24)
                )
                if word & 0x80000000:
                    word -= 0x100000000

                # SPH0645 typically provides left-justified signed audio in 32-bit words.
                # Shift down to signed 16-bit to keep preprocessing memory-friendly.
                sample = word >> 14
                if sample > 32767:
                    sample = 32767
                elif sample < -32768:
                    sample = -32768

                target_samples[sample_index] = sample
                sample_index += 1
                if sample_index >= total_samples:
                    break


def create_sample_buffer():
    return array("h", [0] * config.NUM_SAMPLES)

