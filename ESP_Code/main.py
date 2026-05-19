import gc
import time

import config
from audio_capture import AudioCapture, create_sample_buffer
from button_handler import ButtonHandler
from classifier import Classifier
from inference import CNNInference
from model_loader import ModelLoader
from output_control import OutputControl
from preprocess import MFCCExtractor


STATE_IDLE = 0
STATE_TRIGGERED = 1
STATE_PROCESSING = 2
STATE_OUTPUT = 3


class App:
    def __init__(self):
        gc.collect()
        self.state = STATE_IDLE
        self.audio = AudioCapture()
        self.samples = create_sample_buffer()
        self.button = ButtonHandler(
            config.BUTTON_PIN,
            config.DEBOUNCE_MS,
            active_low=bool(config.BUTTON_ACTIVE_LOW),
        )
        self.mfcc = MFCCExtractor()
        self.outputs = OutputControl()
        self.classifier = Classifier(self.outputs)
        self.model = ModelLoader(config.MODEL_PATH).load()
        self.inference = CNNInference(self.model)
        self.last_class = None
        self.last_score = 0.0
        gc.collect()

    def run(self):
        while True:
            if self.state == STATE_IDLE:
                if self.button.update():
                    self.state = STATE_TRIGGERED
                else:
                    time.sleep_ms(10)

            elif self.state == STATE_TRIGGERED:
                self.button.set_enabled(False)
                gc.collect()
                self.audio.capture_into(self.samples)
                self.state = STATE_PROCESSING

            elif self.state == STATE_PROCESSING:
                mfcc_matrix = self.mfcc.extract(self.samples)
                self.last_class, self.last_score, _ = self.inference.forward(mfcc_matrix)
                print("class=", self.last_class, "label=", config.CLASS_LABELS.get(self.last_class, "?"), "score=", self.last_score)
                self.state = STATE_OUTPUT
                gc.collect()

            elif self.state == STATE_OUTPUT:
                self.classifier.execute(self.last_class)
                self.button.set_enabled(True)
                self.state = STATE_IDLE


def main():
    app = App()
    app.run()


main()
