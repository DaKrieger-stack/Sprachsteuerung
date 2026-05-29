import json
from array import array

import config


EMBEDDED_TEMPLATES = {
    config.COMMAND_ON: [],
    config.COMMAND_OFF: [],
}


class TemplateStore:
    def __init__(self, path=config.TEMPLATE_STORE_PATH):
        self.path = path

    def _feature_shape(self):
        return [config.FRAME_COUNT, config.NUM_MELS]

    def _new_template_dict(self):
        return {
            "version": 1,
            "shape": self._feature_shape(),
            "templates": {
                config.COMMAND_ON: [],
                config.COMMAND_OFF: [],
            },
        }

    def _copy_embedded(self):
        data = self._new_template_dict()
        for label in config.COMMANDS:
            for values in EMBEDDED_TEMPLATES.get(label, ()):
                data["templates"][label].append([float(v) for v in values])
        return data

    def load_raw(self):
        try:
            with open(self.path, "r") as handle:
                data = json.load(handle)
        except OSError:
            data = self._copy_embedded()

        if data.get("shape") != self._feature_shape():
            raise ValueError("Template feature shape does not match current config")

        templates = data.get("templates", {})
        for label in config.COMMANDS:
            templates.setdefault(label, [])
        data["templates"] = templates
        return data

    def load(self):
        raw = self.load_raw()
        result = {}
        for label in config.COMMANDS:
            result[label] = []
            for values in raw["templates"].get(label, ()):
                result[label].append(array("f", values))
        return result

    def save_raw(self, data):
        with open(self.path, "w") as handle:
            json.dump(data, handle)

    def save_templates(self, templates_by_label):
        data = self._new_template_dict()
        for label in config.COMMANDS:
            for template in templates_by_label.get(label, ()):
                data["templates"][label].append([round(float(v), 6) for v in template])
        self.save_raw(data)

    def clear(self, label=None):
        data = self.load_raw()
        if label is None:
            for command in config.COMMANDS:
                data["templates"][command] = []
        else:
            data["templates"][label] = []
        self.save_raw(data)

    def add_template(self, label, feature_vector):
        data = self.load_raw()
        templates = data["templates"][label]
        templates.append([round(float(v), 6) for v in feature_vector])
        if len(templates) > config.MAX_TEMPLATES_PER_CLASS:
            del templates[0]
        self.save_raw(data)

    def average_class(self, label):
        data = self.load_raw()
        templates = data["templates"].get(label, [])
        if not templates:
            return False

        length = len(templates[0])
        avg = [0.0] * length
        for template in templates:
            for i in range(length):
                avg[i] += template[i]

        inv_count = 1.0 / len(templates)
        for i in range(length):
            avg[i] = round(avg[i] * inv_count, 6)

        data["templates"][label] = [avg]
        self.save_raw(data)
        return True

