# -*- coding: utf-8 -*-
"""
PipelineGraph: verwaltet Knoten, Positionen und Kanten (gerichtet),
unterstuetzt Topological Sort und Ausfuehrung im UI.
"""

from typing import Dict, List, Tuple


class _IDGen:
    def __init__(self):
        self._i = 0

    def next(self):
        self._i += 1
        return self._i


class PipelineGraph:
    def __init__(self):
        self.nodes: Dict[int, BaseNode] = {}
        self.edges: List[Tuple[int, int]] = []
        self._idgen = _IDGen()

    def add_node(self, node: "BaseNode") -> "BaseNode":
        node.id = self._idgen.next()
        self.nodes[node.id] = node
        return node

    def add_edge(self, src_id: int, dst_id: int):
        if src_id not in self.nodes or dst_id not in self.nodes:
            raise ValueError("Ungueltige Knoten-ID")
        if src_id == dst_id:
            raise ValueError("Selbst-Verbindungen sind nicht erlaubt")
        if (src_id, dst_id) in self.edges:
            return

        self.edges.append((src_id, dst_id))
        try:
            _ = self.topological_order()
        except ValueError:
            self.edges.pop()
            raise ValueError("Die Verbindung erzeugt einen Zyklus - nicht erlaubt")

    def remove_node(self, node_id: int):
        if node_id not in self.nodes:
            return
        del self.nodes[node_id]
        self.edges = [(src, dst) for (src, dst) in self.edges if src != node_id and dst != node_id]

    def predecessors(self, node_id: int) -> List[int]:
        return [src for (src, dst) in self.edges if dst == node_id]

    def successors(self, node_id: int) -> List[int]:
        return [dst for (src, dst) in self.edges if src == node_id]

    def topological_order(self) -> List[int]:
        indeg = {nid: 0 for nid in self.nodes}
        for _, dst in self.edges:
            indeg[dst] += 1

        queue = [nid for nid, deg in indeg.items() if deg == 0]
        order = []
        edges = list(self.edges)

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for edge in list(edges):
                src, dst = edge
                if src == nid:
                    edges.remove(edge)
                    indeg[dst] -= 1
                    if indeg[dst] == 0:
                        queue.append(dst)

        if edges:
            raise ValueError("Zyklische Abhaengigkeit in der Pipeline")
        return order


class BaseNode:
    """Abstrakte Basisklasse fuer Pipeline-Knoten."""

    def __init__(self):
        self.id = None
        self.pos = (50, 50)
        self.parameters = {}
        self.functions = {}
        self.last_data = None
        self.num_inputs = 1
        self.num_outputs = 1

    def display_name(self) -> str:
        if self.parameters and "name" in self.parameters:
            return self.parameters["name"]["value"]
        return self.__class__.__name__

    def process(self, data):
        self.last_data = data
        return data

    def process_list(self, inputs: List[object]):
        data = inputs[0] if inputs else None
        self.last_data = data
        return data
