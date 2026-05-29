import math
from array import array

import config


class LogMelExtractor:
    def __init__(self):
        self.frame_len = config.FRAME_LEN
        self.hop_len = config.HOP_LEN
        self.n_fft = config.N_FFT
        self.num_mels = config.NUM_MELS
        self.frame_count = config.FRAME_COUNT
        self.pre_emphasis = config.PRE_EMPHASIS
        self.window = self._build_hamming_window(self.frame_len)
        self.frame_buffer = array("f", [0.0] * self.n_fft)
        self.imag_buffer = array("f", [0.0] * self.n_fft)
        self.power_buffer = array("f", [0.0] * ((self.n_fft // 2) + 1))
        self.mel_buffer = array("f", [0.0] * self.num_mels)
        self._twiddle_cos = array("f", [0.0] * (self.n_fft // 2))
        self._twiddle_sin = array("f", [0.0] * (self.n_fft // 2))
        self._build_twiddles()
        self.mel_filters = self._build_mel_filterbank()
        self.output = [array("f", [0.0] * self.num_mels) for _ in range(self.frame_count)]

    def _build_hamming_window(self, length):
        window = array("f", [0.0] * length)
        denom = float(length - 1)
        for i in range(length):
            window[i] = 0.54 - 0.46 * math.cos((2.0 * math.pi * i) / denom)
        return window

    def _build_twiddles(self):
        for k in range(self.n_fft // 2):
            angle = -2.0 * math.pi * k / self.n_fft
            self._twiddle_cos[k] = math.cos(angle)
            self._twiddle_sin[k] = math.sin(angle)

    def _hz_to_mel(self, hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700.0 * ((10.0 ** (mel / 2595.0)) - 1.0)

    def _build_mel_filterbank(self):
        min_mel = self._hz_to_mel(config.MEL_MIN_HZ)
        max_mel = self._hz_to_mel(config.MEL_MAX_HZ)
        mel_points = [0.0] * (self.num_mels + 2)
        bin_points = [0] * (self.num_mels + 2)
        mel_step = (max_mel - min_mel) / (self.num_mels + 1)

        for i in range(self.num_mels + 2):
            mel_points[i] = min_mel + (i * mel_step)
            hz = self._mel_to_hz(mel_points[i])
            index = int((self.n_fft + 1) * hz / config.SAMPLE_RATE)
            if index < 0:
                index = 0
            elif index > self.n_fft // 2:
                index = self.n_fft // 2
            bin_points[i] = index

        filters = []
        for m in range(1, self.num_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]

            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1

            weights = []
            for k in range(left, center):
                weights.append((k, (k - left) / float(center - left)))
            for k in range(center, right):
                weights.append((k, (right - k) / float(right - center)))
            filters.append(weights)
        return filters

    def _fft_inplace(self):
        n = self.n_fft
        real = self.frame_buffer
        imag = self.imag_buffer

        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                tmp = real[i]
                real[i] = real[j]
                real[j] = tmp
                tmp = imag[i]
                imag[i] = imag[j]
                imag[j] = tmp

        size = 2
        while size <= n:
            half = size >> 1
            step = n // size
            for start in range(0, n, size):
                tw = 0
                for offset in range(half):
                    match = start + offset + half
                    cosv = self._twiddle_cos[tw]
                    sinv = self._twiddle_sin[tw]
                    tr = (real[match] * cosv) - (imag[match] * sinv)
                    ti = (real[match] * sinv) + (imag[match] * cosv)
                    ur = real[start + offset]
                    ui = imag[start + offset]
                    real[match] = ur - tr
                    imag[match] = ui - ti
                    real[start + offset] = ur + tr
                    imag[start + offset] = ui + ti
                    tw += step
            size <<= 1

    def _fill_frame(self, samples, start_index):
        prev = samples[start_index]
        self.frame_buffer[0] = prev * self.window[0]
        self.imag_buffer[0] = 0.0

        for i in range(1, self.frame_len):
            current = samples[start_index + i]
            emphasized = current - (self.pre_emphasis * prev)
            self.frame_buffer[i] = emphasized * self.window[i]
            self.imag_buffer[i] = 0.0
            prev = current

        for i in range(self.frame_len, self.n_fft):
            self.frame_buffer[i] = 0.0
            self.imag_buffer[i] = 0.0

    def _power_spectrum(self):
        for i in range((self.n_fft // 2) + 1):
            real = self.frame_buffer[i]
            imag = self.imag_buffer[i]
            self.power_buffer[i] = (real * real + imag * imag) / self.n_fft

    def _apply_mel_filterbank(self, out_vec):
        for m in range(self.num_mels):
            acc = 0.0
            filt = self.mel_filters[m]
            for pair in filt:
                acc += self.power_buffer[pair[0]] * pair[1]
            if acc < 1e-10:
                acc = 1e-10
            out_vec[m] = math.log(acc)

    def _normalize_inplace(self):
        total = 0.0
        count = self.frame_count * self.num_mels
        for frame in self.output:
            for value in frame:
                total += value
        mean = total / count

        variance = 0.0
        for frame in self.output:
            for value in frame:
                delta = value - mean
                variance += delta * delta
        std = math.sqrt(variance / count)
        if std < 1e-6:
            std = 1.0

        inv_std = 1.0 / std
        for frame in self.output:
            for i in range(self.num_mels):
                frame[i] = (frame[i] - mean) * inv_std

    def extract(self, samples):
        for frame_idx in range(self.frame_count):
            start = frame_idx * self.hop_len
            self._fill_frame(samples, start)
            self._fft_inplace()
            self._power_spectrum()
            self._apply_mel_filterbank(self.output[frame_idx])
        self._normalize_inplace()
        return self.output


def flatten_feature_matrix(matrix):
    flat = array("f", [0.0] * (len(matrix) * len(matrix[0])))
    idx = 0
    for row in matrix:
        for value in row:
            flat[idx] = value
            idx += 1
    return flat


def rms_level(samples):
    acc = 0.0
    for sample in samples:
        acc += sample * sample
    return math.sqrt(acc / len(samples))

