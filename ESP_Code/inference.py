import math
from array import array


class CNNInference:
    def __init__(self, model):
        self.layers = model["layers"]
        self.input_shape = model["input_shape"]

    def _relu_inplace(self, tensor):
        for i in range(len(tensor)):
            if tensor[i] < 0.0:
                tensor[i] = 0.0

    def _softmax(self, logits):
        max_val = logits[0]
        for value in logits:
            if value > max_val:
                max_val = value

        total = 0.0
        for i in range(len(logits)):
            logits[i] = math.exp(logits[i] - max_val)
            total += logits[i]

        inv_total = 1.0 / total
        for i in range(len(logits)):
            logits[i] *= inv_total
        return logits

    def _conv2d(self, input_tensor, shape, layer):
        in_h, in_w, in_c = shape
        kernel_h = layer["kernel_h"]
        kernel_w = layer["kernel_w"]
        stride_h = layer["stride_h"]
        stride_w = layer["stride_w"]
        out_c = layer["out_channels"]
        padding = layer["padding"]

        if padding:
            out_h = (in_h + stride_h - 1) // stride_h
            out_w = (in_w + stride_w - 1) // stride_w
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
        else:
            out_h = ((in_h - kernel_h) // stride_h) + 1
            out_w = ((in_w - kernel_w) // stride_w) + 1
            pad_h = 0
            pad_w = 0

        output = array("f", [0.0] * (out_h * out_w * out_c))
        weights = layer["weights"]
        biases = layer["biases"]

        out_index = 0
        for oy in range(out_h):
            base_y = oy * stride_h
            for ox in range(out_w):
                base_x = ox * stride_w
                for oc in range(out_c):
                    acc = biases[oc]
                    for ky in range(kernel_h):
                        in_y = base_y + ky - pad_h
                        if in_y < 0 or in_y >= in_h:
                            continue
                        for kx in range(kernel_w):
                            in_x = base_x + kx - pad_w
                            if in_x < 0 or in_x >= in_w:
                                continue
                            input_base = ((in_y * in_w) + in_x) * in_c
                            weight_base = ((((oc * kernel_h) + ky) * kernel_w) + kx) * in_c
                            for ic in range(in_c):
                                acc += input_tensor[input_base + ic] * weights[weight_base + ic]
                    output[out_index] = acc
                    out_index += 1

        self._relu_inplace(output)
        return output, (out_h, out_w, out_c)

    def _maxpool2d(self, input_tensor, shape, layer):
        in_h, in_w, in_c = shape
        pool_h = layer["pool_h"]
        pool_w = layer["pool_w"]
        stride_h = layer["stride_h"]
        stride_w = layer["stride_w"]

        out_h = ((in_h - pool_h) // stride_h) + 1
        out_w = ((in_w - pool_w) // stride_w) + 1
        output = array("f", [0.0] * (out_h * out_w * in_c))

        out_index = 0
        for oy in range(out_h):
            for ox in range(out_w):
                for ic in range(in_c):
                    max_val = -1e30
                    for ky in range(pool_h):
                        in_y = oy * stride_h + ky
                        for kx in range(pool_w):
                            in_x = ox * stride_w + kx
                            idx = ((in_y * in_w) + in_x) * in_c + ic
                            value = input_tensor[idx]
                            if value > max_val:
                                max_val = value
                    output[out_index] = max_val
                    out_index += 1

        return output, (out_h, out_w, in_c)

    def _dense(self, input_tensor, layer):
        out_features = layer["out_features"]
        in_features = layer["in_features"]
        weights = layer["weights"]
        biases = layer["biases"]
        output = array("f", [0.0] * out_features)

        for out_idx in range(out_features):
            acc = biases[out_idx]
            weight_offset = out_idx * in_features
            for in_idx in range(in_features):
                acc += input_tensor[in_idx] * weights[weight_offset + in_idx]
            output[out_idx] = acc
        return output

    def forward(self, mfcc_matrix):
        frame_count = len(mfcc_matrix)
        coeff_count = len(mfcc_matrix[0])
        expected_h, expected_w, expected_c = self.input_shape
        if frame_count != expected_h or coeff_count != expected_w or expected_c != 1:
            raise ValueError("MFCC input shape does not match model header")
        input_tensor = array("f", [0.0] * (frame_count * coeff_count))

        idx = 0
        for frame in mfcc_matrix:
            for coeff in frame:
                input_tensor[idx] = coeff
                idx += 1

        current = input_tensor
        shape = (frame_count, coeff_count, 1)

        for layer in self.layers:
            layer_type = layer["type"]
            if layer_type == "conv2d":
                current, shape = self._conv2d(current, shape, layer)
            elif layer_type == "maxpool2d":
                current, shape = self._maxpool2d(current, shape, layer)
            elif layer_type == "flatten":
                shape = (1, 1, len(current))
            elif layer_type == "dense":
                current = self._dense(current, layer)
                shape = (1, 1, len(current))
            else:
                raise ValueError("unsupported layer")

        probabilities = self._softmax(current)
        best_idx = 0
        best_prob = probabilities[0]
        for i in range(1, len(probabilities)):
            if probabilities[i] > best_prob:
                best_prob = probabilities[i]
                best_idx = i
        return best_idx, best_prob, probabilities
