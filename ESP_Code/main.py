import gc
import time

import config
from audio_capture import AudioCapture
from classifier import TemplateClassifier
from output_control import OutputControl
from preprocess import LogMelExtractor, flatten_feature_matrix, rms_level
from template_store import TemplateStore


class VoiceCommandApp:
    def __init__(self):
        gc.collect()
        self.audio = AudioCapture()
        self.extractor = LogMelExtractor()
        self.outputs = OutputControl()
        self.store = TemplateStore()
        self.templates = self.store.load()
        self.classifier = TemplateClassifier(self.templates)
        gc.collect()

    def reload_templates(self):
        self.templates = self.store.load()
        self.classifier.update_templates(self.templates)

    def capture_features(self):
        samples = self.audio.read_frame()
        level = rms_level(samples)
        matrix = self.extractor.extract(samples)
        vector = flatten_feature_matrix(matrix)
        return level, matrix, vector

    def infer_once(self, verbose=True):
        level, _, vector = self.capture_features()
        if level < config.MIN_SIGNAL_RMS:
            if verbose:
                print("ignored: low signal, rms=", level)
            return None

        result = self.classifier.classify(vector)
        label = result["label"]
        if label is not None:
            self.outputs.apply(label)
            if verbose:
                print(
                    "detected=",
                    label,
                    "score=",
                    result["best_score"],
                    "margin=",
                    result["margin"],
                    "licht=",
                    self.outputs.state(),
                )
        elif verbose:
            print(
                "ignored: no confident match, scores=",
                result["class_scores"],
                "margin=",
                result["margin"],
            )
        return result

    def listen_forever(self):
        print("voice loop active")
        print("templates on=", len(self.templates[config.COMMAND_ON]), "off=", len(self.templates[config.COMMAND_OFF]))
        while True:
            self.infer_once(verbose=True)
            gc.collect()

    def record_template(self, label, countdown_s=2):
        print("prepare sample for", label)
        for remaining in range(countdown_s, 0, -1):
            print("recording starts in", remaining)
            time.sleep(1)

        level, _, vector = self.capture_features()
        if level < config.MIN_SIGNAL_RMS:
            print("sample rejected, rms too low:", level)
            return False

        self.store.add_template(label, vector)
        self.reload_templates()
        print("stored template for", label, "rms=", level, "count=", len(self.templates[label]))
        return True

    def average_templates(self, label):
        if self.store.average_class(label):
            self.reload_templates()
            print("averaged templates for", label)
            return True
        print("no templates available for", label)
        return False

    def clear_templates(self, label=None):
        self.store.clear(label)
        self.reload_templates()
        if label is None:
            print("cleared all templates")
        else:
            print("cleared templates for", label)

    def train_interactive(self):
        print("training mode")
        print("commands: on, off, avg_on, avg_off, clear_on, clear_off, clear_all, listen, exit")
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "on":
                self.record_template(config.COMMAND_ON)
            elif cmd == "off":
                self.record_template(config.COMMAND_OFF)
            elif cmd == "avg_on":
                self.average_templates(config.COMMAND_ON)
            elif cmd == "avg_off":
                self.average_templates(config.COMMAND_OFF)
            elif cmd == "clear_on":
                self.clear_templates(config.COMMAND_ON)
            elif cmd == "clear_off":
                self.clear_templates(config.COMMAND_OFF)
            elif cmd == "clear_all":
                self.clear_templates()
            elif cmd == "listen":
                self.listen_forever()
            elif cmd == "exit":
                break
            else:
                print("unknown command")


def main(training=False):
    app = VoiceCommandApp()
    if training:
        app.train_interactive()
    else:
        app.listen_forever()


if __name__ == "__main__":
    main()
