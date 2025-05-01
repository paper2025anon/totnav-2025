"""
TurnAnalyzer Module for TOT-NAV

Provides:
  - Turn labeling (including slight-turns)
  - Feasibility checks for user-requested turns
  - Penalty computation for route scoring
  - Generation of candidate-route summaries for LLM evaluation
"""

from typing import Any, Dict, List

from tot_nav.graph_loader import GraphLoader
from tot_nav.utils import (
    Turn,
    classify_three_nodes,
    classify_initial_heading,
    DistanceUtils
)


class TurnAnalyzer:
    def __init__(
            self,
            graph_loader: GraphLoader,
            distance_utils: DistanceUtils,
            alpha: float = 1.0,
            beta: float = 2.0,
            gamma: float = 3.0
    ):
        """
        Initialize the turn analyzer.

        Args:
            graph_loader: Provides graph with node positions & neighbors
            distance_utils: Provides shortest-path & distance computations
            alpha: Penalty for low-degree node when a non-straight turn is requested
            beta: Penalty when requested turn not available at all
            gamma: Penalty when chosen edge violates requested turn
        """
        self.gl = graph_loader
        self.du = distance_utils
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def label_turn(self, prev: Any, curr: Any, nxt: Any) -> Turn:
        """
        Classify actual turn at `curr` for edge curr→nxt.
        - If prev is None, compare curr→nxt to +X axis.
        - Else use prev→curr→nxt classification.
        """
        curr_pos = self.gl.graph.nodes[curr]["position"]
        next_pos = self.gl.graph.nodes[nxt]["position"]
        if prev is None:
            return classify_initial_heading(curr_pos, next_pos)
        prev_pos = self.gl.graph.nodes[prev]["position"]
        return classify_three_nodes(prev_pos, curr_pos, next_pos)

    def neighbors_by_turn(self, prev: Any, curr: Any) -> Dict[Turn, List[Any]]:
        """
        Bucket each neighbor of `curr` by its actual turn label relative to `prev`.
        Returns a dict: {Turn.STRAIGHT: [n1, n2], Turn.LEFT: [n3], ...}
        """
        buckets: Dict[Turn, List[Any]] = {t: [] for t in Turn}
        for nb in self.gl.graph.neighbors(curr):
            label = self.label_turn(prev, curr, nb)
            buckets[label].append(nb)
        return buckets

    def is_turn_feasible(self, prev: Any, curr: Any, required: Turn) -> bool:
        """
        True if any outgoing edge from `curr` realizes the `required` turn.
        """
        return len(self.neighbors_by_turn(prev, curr).get(required, [])) > 0

    def continuity_violation(self, prev: Any, curr: Any, nxt: Any, required: Turn) -> bool:
        """
        True if taking edge curr→nxt does NOT realize the `required` turn.
        """
        actual = self.label_turn(prev, curr, nxt)
        return actual is not required

    def turn_penalty(self, prev: Any, curr: Any, nxt: Any, required: Turn) -> float:
        """
        Penalty p_t when at `curr` aiming for `nxt` but `required` was requested:
          1) if deg(curr)<3 and required is LEFT/RIGHT: α
          2) elif required not feasible at all: β
          3) elif chosen nxt violates required: γ
          else 0.
        """
        deg = self.gl.graph.degree[curr]
        if deg < 3 and required in (Turn.LEFT, Turn.RIGHT):
            return self.alpha
        if not self.is_turn_feasible(prev, curr, required):
            return self.beta
        if self.continuity_violation(prev, curr, nxt, required):
            return self.gamma
        return 0.0

    def choose_next(self, prev: Any, curr: Any, required: Turn) -> List[Any]:
        """
        Return neighbors of `curr` ordered so that:
          1) those matching `required` first,
          2) if required=STRAIGHT, slight deviations next,
          3) then all others.
        """
        buckets = self.neighbors_by_turn(prev, curr)
        ordered: List[Any] = []
        # exact matches
        ordered += buckets.get(required, [])
        # slight deviations if straight
        if required is Turn.STRAIGHT:
            for sl in (Turn.SLIGHT_LEFT, Turn.SLIGHT_RIGHT):
                ordered += buckets.get(sl, [])
        # the rest
        for label, nbs in buckets.items():
            if label is required or (required is Turn.STRAIGHT and label in (Turn.SLIGHT_LEFT, Turn.SLIGHT_RIGHT)):
                continue
            ordered += nbs
        return ordered

    def compute_candidate_paths(
            self,
            start: Any,
            targets: List[Any]
    ) -> Dict[Any, List[Any]]:
        """
        For each target node, return the shortest-path node sequence from start.
        """
        paths: Dict[Any, List[Any]] = {}
        for t in targets:
            p = self.du.get_shortest_path(start, t)
            if p:
                paths[t] = p
        return paths

    def get_path_info(self, path: List[Any]) -> Dict[str, Any]:
        """
        Given a node sequence, compute:
          - 'distance': total Euclidean length
          - 'turns': List[Turn] at each step
        """
        dist = self.du.get_path_distance(path)
        turns: List[Turn] = []
        for i in range(len(path) - 1):
            prev = None if i == 0 else path[i - 1]
            curr, nxt = path[i], path[i + 1]
            turns.append(self.label_turn(prev, curr, nxt))
        return {"distance": dist, "turns": turns}

    def get_available_turns_at(self, path: List[Any]) -> Dict[str, List[Any]]:
        """
        For the last node in path, bucket its neighbors by actual turn
        and return as {turn_value: [node_ids]}.
        """
        if len(path) < 2:
            prev = None
        else:
            prev = path[-2]
        curr = path[-1]
        buckets = self.neighbors_by_turn(prev, curr)
        return {t.value: buckets[t] for t in buckets if buckets[t]}

    def summarize_candidate(self, path: List[Any], landmark: str) -> Dict[str, Any]:
        """
        Build a summary for LLM input:
          - distance (float)
          - path_str ("10 → 2 → 3")
          - turns_str ("10→2: STRAIGHT, 2→3: LEFT")
          - at_node (last node)
          - landmark (name)
          - available_turns ({ "LEFT": [5], "STRAIGHT": [4,6], ... })
        """
        info = self.get_path_info(path)
        distance = info["distance"]

        # path string
        path_str = " → ".join(str(n) for n in path)

        # turns string
        turns_str = ", ".join(
            f"{path[i]}→{path[i + 1]}: {info['turns'][i].value}"
            for i in range(len(path) - 1)
        )

        at_node = path[-1]
        avail = self.get_available_turns_at(path)

        return {
            "distance": distance,
            "path_str": path_str,
            "turns_str": turns_str,
            "at_node": at_node,
            "landmark": landmark,
            "available_turns": avail
        }