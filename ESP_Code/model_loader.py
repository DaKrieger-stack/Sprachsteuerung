import gc
import struct
from array import array


LAYER_CONV2D = 1
LAYER_MAXPOOL2D = 2
LAYER_DENSE = 3
LAYER_FLATTEN = 4


class ModelLoader:
    def __init__(self, path):
        self.path = path

    def _read_exact(self, handle, size):
        data = handle.read(size)
        if data is None or len(data) != size:
            raise ValueError("model.dat truncated")
        return data

    def _read_floats(self, handle, count):
        buf = self._read_exact(handle, count * 4)
        values = array("f", [0.0] * count)
        offset = 0
        for i in range(count):
            values[i] = struct.unpack_from("<f", buf, offset)[0]
            offset += 4
        return values

    def load(self):
        with open(self.path, "rb") as handle:
            magic = self._read_exact(handle, 4)
            if magic != b"MCNN":
                raise ValueError("unsupported model format")

            version = self._read_exact(handle, 1)[0]
            if version != 1:
                raise ValueError("unsupported model version")

            input_h, input_w, input_c, layer_count = struct.unpack(
                "<HHHB", self._read_exact(handle, 7)
            )
            expected_input_shape = (input_h, input_w, input_c)
            layers = []

            for _ in range(layer_count):
                layer_type = self._read_exact(handle, 1)[0]
                if layer_type == LAYER_CONV2D:
                    out_c, kernel_h, kernel_w, stride_h, stride_w, padding = struct.unpack(
                        "<HHHHHB", self._read_exact(handle, 11)
                    )
                    weight_count = input_c * out_c * kernel_h * kernel_w
                    bias_count = out_c
                    weights = self._read_floats(handle, weight_count)
                    biases = self._read_floats(handle, bias_count)
                    layers.append(
                        {
                            "type": "conv2d",
                            "in_channels": input_c,
                            "out_channels": out_c,
                            "kernel_h": kernel_h,
                            "kernel_w": kernel_w,
                            "stride_h": stride_h,
                            "stride_w": stride_w,
                            "padding": padding,
                            "weights": weights,
                            "biases": biases,
                        }
                    )
                    if padding:
                        input_h = (input_h + stride_h - 1) // stride_h
                        input_w = (input_w + stride_w - 1) // stride_w
                    else:
                        input_h = ((input_h - kernel_h) // stride_h) + 1
                        input_w = ((input_w - kernel_w) // stride_w) + 1
                    input_c = out_c

                elif layer_type == LAYER_MAXPOOL2D:
                    pool_h, pool_w, stride_h, stride_w = struct.unpack(
                        "<HHHH", self._read_exact(handle, 8)
                    )
                    layers.append(
                        {
                            "type": "maxpool2d",
                            "pool_h": pool_h,
                            "pool_w": pool_w,
                            "stride_h": stride_h,
                            "stride_w": stride_w,
                        }
                    )
                    input_h = ((input_h - pool_h) // stride_h) + 1
                    input_w = ((input_w - pool_w) // stride_w) + 1

                elif layer_type == LAYER_FLATTEN:
                    layers.append({"type": "flatten"})
                    flattened = input_h * input_w * input_c
                    input_h = 1
                    input_w = 1
                    input_c = flattened

                elif layer_type == LAYER_DENSE:
                    out_features = struct.unpack("<H", self._read_exact(handle, 2))[0]
                    in_features = input_h * input_w * input_c
                    weight_count = in_features * out_features
                    bias_count = out_features
                    weights = self._read_floats(handle, weight_count)
                    biases = self._read_floats(handle, bias_count)
                    layers.append(
                        {
                            "type": "dense",
                            "in_features": in_features,
                            "out_features": out_features,
                            "weights": weights,
                            "biases": biases,
                        }
                    )
                    input_h = 1
                    input_w = 1
                    input_c = out_features
                else:
                    raise ValueError("unknown layer type")

                gc.collect()

        return {
            "input_shape": expected_input_shape,
            "layers": layers,
        }
