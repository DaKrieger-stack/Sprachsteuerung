# -*- coding: utf-8 -*-
"""
Basisklasse fuer Filter (Komponententyp in der Pipeline).
Erbt von BaseNode und stellt Hilfsmethoden bereit.
"""

from pipeline.graph import BaseNode


class BaseFilter(BaseNode):
    def __init__(self):
        super().__init__()
        self.num_inputs = 1
        self.num_outputs = 1

    def get(self, key, default=None):
        meta = self.parameters.get(key)
        return meta["value"] if meta else default
