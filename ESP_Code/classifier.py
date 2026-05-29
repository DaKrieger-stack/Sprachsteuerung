import math

import config


class TemplateClassifier:
    def __init__(self, templates_by_label):
        self.templates_by_label = templates_by_label

    def update_templates(self, templates_by_label):
        self.templates_by_label = templates_by_label

    def _cosine_similarity(self, a, b):
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(len(a)):
            av = a[i]
            bv = b[i]
            dot += av * bv
            norm_a += av * av
            norm_b += bv * bv
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        return dot / math.sqrt(norm_a * norm_b)

    def _best_score_for_label(self, feature_vector, label):
        templates = self.templates_by_label.get(label, ())
        best_score = -1.0
        best_index = -1
        for idx in range(len(templates)):
            score = self._cosine_similarity(feature_vector, templates[idx])
            if score > best_score:
                best_score = score
                best_index = idx
        return best_score, best_index

    def classify(self, feature_vector):
        class_scores = {}
        best_label = None
        best_score = -1.0
        second_best = -1.0
        best_template_index = -1

        for label in config.COMMANDS:
            score, template_index = self._best_score_for_label(feature_vector, label)
            class_scores[label] = score
            if score > best_score:
                second_best = best_score
                best_score = score
                best_label = label
                best_template_index = template_index
            elif score > second_best:
                second_best = score

        margin = best_score - second_best
        accepted = (
            best_label is not None
            and best_score >= config.SIMILARITY_THRESHOLD
            and margin >= config.MARGIN_THRESHOLD
        )

        if not accepted:
            best_label = None

        return {
            "label": best_label,
            "best_score": best_score,
            "margin": margin,
            "best_template_index": best_template_index,
            "class_scores": class_scores,
        }

