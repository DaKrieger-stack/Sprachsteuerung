from array import array
from machine import I2S, Pin

import config


class AudioCapture:
    def __init__(self):
        bytes_per_sample = 4 if config.I2S_BITS > 16 else 2
        self._raw_chunk = bytearray(config.CHUNK_SAMPLES * bytes_per_sample)
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

    def _decode_sample(self, raw, base):
        if config.I2S_BITS > 16:
            word = (
                raw[base]
                | (raw[base + 1] << 8)
                | (raw[base + 2] << 16)
                | (raw[base + 3] << 24)
            )
            if word & 0x80000000:
                word -= 0x100000000
            sample = word >> 14
        else:
            sample = raw[base] | (raw[base + 1] << 8)
            if sample & 0x8000:
                sample -= 0x10000

        if sample > 32767:
            return 32767
        if sample < -32768:
            return -32768
        return sample

    def capture_into(self, target_samples):
        sample_index = 0
        bytes_per_sample = 4 if config.I2S_BITS > 16 else 2
        raw = self._raw_chunk

        while sample_index < len(target_samples):
            bytes_read = self._i2s.readinto(raw)
            if not bytes_read:
                continue

            limit = bytes_read // bytes_per_sample
            for i in range(limit):
                target_samples[sample_index] = self._decode_sample(raw, i * bytes_per_sample)
                sample_index += 1
                if sample_index >= len(target_samples):
                    break

    def read_frame(self):
        samples = create_sample_buffer()
        self.capture_into(samples)
        return samples


def create_sample_buffer():
    return array("h", [0] * config.NUM_SAMPLES)

