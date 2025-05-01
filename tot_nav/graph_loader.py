"""
GraphLoader Module for TOT-NAV
Loads a topological graph and associated node images from a pickle file,
and provides simple accessors for node data and shortest paths.
"""

import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class GraphLoader:
    def __init__(self, pkl_path: str):
        """
        Initialize the GraphLoader.

        Args:
            pkl_path: Path to the pickle file containing graph data.
        """
        self.pkl_path = Path(pkl_path)
        self.graph: Optional[nx.Graph] = None
        self.node_images: Dict[int, List[Any]] = {}
        self.load_graph()

    def load_graph(self) -> None:
        """
        Load the graph structure, node positions, and images from the pickle.
        Expects a dict with keys:
          - 'json_graph': node-link data for networkx.node_link_graph
          - 'pos': list of (x, y) positions, one per node index
          - 'images': list of lists of raw image bytes per node index
        """
        try:
            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)
            # Build graph
            if "json_graph" in data:
                self.graph = nx.node_link_graph(data["json_graph"])
            else:
                raise KeyError("Pickle missing 'json_graph' field")
            # Attach positions
            if "pos" in data:
                positions = {i: tuple(pos) for i, pos in enumerate(data["pos"])}
                nx.set_node_attributes(self.graph, positions, "position")
            # Process images
            if "images" in data:
                self._process_images(data["images"])
            logger.info(
                f"Loaded graph ({self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges)"
            )
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            raise

    def _process_images(self, raw_images: List[List[bytes]]) -> None:
        """
        Decode and store image arrays for each node.

        Args:
            raw_images: list where each element is a list of raw image bytes for a node
        """
        for node_idx, images in enumerate(raw_images):
            decoded = []
            for img_bytes in images:
                # Convert bytes to numpy array, then decode with OpenCV
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded.append(img)
            self.node_images[node_idx] = decoded

    def get_node_data(self, node_id: int) -> Dict[str, Any]:
        """
        Return data for a given node.

        Returns dict with:
          - node_id
          - neighbors: list of neighbor node IDs
          - position: (x, y) or None
          - images: list of decoded images
        """
        if self.graph is None or node_id not in self.graph:
            raise KeyError(f"Node {node_id} not found")
        return {
            "node_id": node_id,
            "neighbors": list(self.graph.neighbors(node_id)),
            "position": self.graph.nodes[node_id].get("position"),
            "images": self.node_images.get(node_id, []),
        }

    def get_node_images(self, node_id: int) -> List[Any]:
        """
        Return list of images for the specified node.
        """
        return self.node_images.get(node_id, [])

    def get_node_neighbors(self, node_id: int) -> List[Dict[str, Any]]:
        """
        Return neighbor info for a node:
          - node_id
          - position
          - images
        """
        if self.graph is None or node_id not in self.graph:
            raise KeyError(f"Node {node_id} not found")
        result = []
        for nbr in self.graph.neighbors(node_id):
            result.append({
                "node_id": nbr,
                "position": self.graph.nodes[nbr].get("position"),
                "images": self.node_images.get(nbr, []),
            })
        return result

    def get_path_between_nodes(
            self, start_id: int, end_id: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Compute shortest path (by hop count) between two nodes.

        Returns a list of node-data dicts along the path, or None if no path exists.
        """
        if self.graph is None:
            raise RuntimeError("Graph not loaded")
        if start_id not in self.graph or end_id not in self.graph:
            raise KeyError("Start or end node not in graph")
        try:
            path = nx.shortest_path(self.graph, start_id, end_id)
            return [self.get_node_data(n) for n in path]
        except nx.NetworkXNoPath:
            return None