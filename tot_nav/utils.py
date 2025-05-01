"""
Utilities for TOT-NAV:
- Distance calculations with Floyd-Warshall algorithm
- Vector math for turn angle classification
- Path distance computation
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class Turn(Enum):
    """Enumeration of possible turn types."""
    STRAIGHT = "STRAIGHT"
    SLIGHT_LEFT = "SLIGHT_LEFT"
    LEFT = "LEFT"
    SLIGHT_RIGHT = "SLIGHT_RIGHT"
    RIGHT = "RIGHT"
    UTURN = "UTURN"


# Vector Math Functions ---------------------------------------------------

def make_vector(
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float]
) -> np.ndarray:
    """
    Return the 2D vector from prev_pos to curr_pos.
    """
    return np.array([curr_pos[0] - prev_pos[0],
                     curr_pos[1] - prev_pos[1]], dtype=float)


def signed_angle(
        v_in: np.ndarray,
        v_out: np.ndarray
) -> float:
    """
    Compute the signed angle (in degrees) from v_in to v_out.
    Positive = left turn, negative = right turn.
    """
    # normalize (avoid division by zero)
    u = v_in / (np.linalg.norm(v_in) + 1e-8)
    w = v_out / (np.linalg.norm(v_out) + 1e-8)
    det = u[0] * w[1] - u[1] * w[0]
    dot = u[0] * w[0] + u[1] * w[1]
    return math.degrees(math.atan2(det, dot))


def classify_turn(angle_deg: float) -> Turn:
    """
    Map a signed angle in degrees to a Turn label, with 'slight' categories:
      |angle| ≤ 15°         → STRAIGHT
      15° < angle ≤ 45°     → SLIGHT_LEFT  (if positive) or SLIGHT_RIGHT (if negative)
      45° < angle < 150°    → LEFT         (if positive) or RIGHT       (if negative)
      |angle| ≥ 150°        → UTURN
    """
    if abs(angle_deg) <= 15.0:
        return Turn.STRAIGHT
    if 15.0 < angle_deg <= 45.0:
        return Turn.SLIGHT_LEFT
    if 45.0 < angle_deg < 150.0:
        return Turn.LEFT
    if -45.0 <= angle_deg < -15.0:
        return Turn.SLIGHT_RIGHT
    if -150.0 < angle_deg < -45.0:
        return Turn.RIGHT
    return Turn.UTURN


def classify_three_nodes(
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        next_pos: Tuple[float, float]
) -> Turn:
    """
    Given three node positions, compute the turn label at curr_pos.
    """
    v_in = make_vector(prev_pos, curr_pos)
    v_out = make_vector(curr_pos, next_pos)
    angle = signed_angle(v_in, v_out)
    return classify_turn(angle)


def classify_initial_heading(
        curr_pos: Tuple[float, float],
        next_pos: Tuple[float, float]
) -> Turn:
    """
    Classify the first segment (starting node → next node) by
    comparing against the global +X axis as the 'incoming' direction.
    """
    v_in = np.array([1.0, 0.0])
    v_out = make_vector(curr_pos, next_pos)
    angle = signed_angle(v_in, v_out)
    return classify_turn(angle)


def discrete_turns() -> List[str]:
    """
    Return the list of all discrete turn labels.
    """
    return [t.value for t in Turn]

class DistanceUtils:
    def __init__(self, graph: nx.Graph):
        """
        Initialize distance utilities for a graph.

        Args:
            graph: A NetworkX graph whose nodes have a 'position' attribute (x, y).
        """
        self.graph = graph
        # dist[u][v] = shortest distance, next_hop[u][v] = the next node on the shortest path
        self.dist: Dict[Any, Dict[Any, float]] = {}
        self.next_hop: Dict[Any, Dict[Any, Optional[Any]]] = {}
        self.compute_floyd_warshall()

    def compute_floyd_warshall(self) -> None:
        """
        Precompute all-pairs shortest distances and next-hop pointers
        using the Floyd–Warshall algorithm.
        """
        nodes = list(self.graph.nodes)
        # initialize
        for u in nodes:
            self.dist[u] = {}
            self.next_hop[u] = {}
            for v in nodes:
                if u == v:
                    self.dist[u][v] = 0.0
                    self.next_hop[u][v] = v
                elif self.graph.has_edge(u, v):
                    # weight = Euclidean distance between positions
                    pos_u = self.graph.nodes[u].get("position")
                    pos_v = self.graph.nodes[v].get("position")
                    if pos_u is None or pos_v is None:
                        w = 1.0
                    else:
                        w = math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])
                    self.dist[u][v] = w
                    self.next_hop[u][v] = v
                else:
                    self.dist[u][v] = float("inf")
                    self.next_hop[u][v] = None

        # core Floyd–Warshall
        for k in nodes:
            for i in nodes:
                # skip unreachable
                if self.dist[i][k] == float("inf"):
                    continue
                for j in nodes:
                    new_d = self.dist[i][k] + self.dist[k][j]
                    if new_d < self.dist[i][j]:
                        self.dist[i][j] = new_d
                        self.next_hop[i][j] = self.next_hop[i][k]

    def get_shortest_path(self, start: Any, end: Any) -> Optional[List[Any]]:
        """
        Reconstruct the node sequence for the shortest path from start to end.
        Returns None if no path exists.
        """
        if self.next_hop.get(start, {}).get(end) is None:
            return None
        path = [start]
        while start != end:
            start = self.next_hop[start][end]
            if start is None:
                return None
            path.append(start)
        return path

    def get_path_distance(self, path: List[Any]) -> float:
        """
        Given a sequence of nodes, sum their Euclidean distances.
        """
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            pos_u = self.graph.nodes[u].get("position")
            pos_v = self.graph.nodes[v].get("position")
            if pos_u is None or pos_v is None:
                total += 1.0
            else:
                total += math.hypot(pos_v[0] - pos_u[0], pos_v[1] - pos_u[1])
        return total