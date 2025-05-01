import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Optional


class RouteVisualizer:
    """
    Visualizes the navigation route with speed-based coloring.
    """

    # Speed color map
    SPEED_COLORS = {
        "VERY_SLOW": "#1E88E5",  # Blue
        "SLOW": "#26A69A",  # Teal
        "NORMAL": "#66BB6A",  # Green
        "FAST": "#FFA726",  # Orange
        "VERY_FAST": "#EF5350",  # Red
        "NOT_SPECIFIED": "#9E9E9E"  # Gray
    }

    def __init__(self, graph_loader, distance_utils):
        """Initialize the route visualizer."""
        self.gl = graph_loader
        self.du = distance_utils

    def visualize_route(
            self,
            route: List[int],
            parsed_instruction: Dict[str, Any],
            output_path: Optional[str] = None,
            title: str = "TOT-NAV Route Visualization",
            show_plot: bool = True
    ):
        """
        Visualize the complete route with speed-based coloring.
        """
        # Create figure
        plt.figure(figsize=(20, 16), dpi=100)

        # Get node positions
        pos = nx.get_node_attributes(self.gl.graph, 'position')
        if not pos:
            pos = nx.spring_layout(self.gl.graph, k=4.0, iterations=300)

        # Scale positions
        pos_2d = {node: np.array([coords[0] * 8.0, coords[1] * 8.0]) for node, coords in pos.items()}

        # Draw graph background
        nx.draw_networkx_edges(
            self.gl.graph, pos_2d,
            edge_color='#DDDDDD',
            alpha=0.4,
            width=1.5
        )

        # Draw all nodes
        all_nodes = list(self.gl.graph.nodes())
        nx.draw_networkx_nodes(
            self.gl.graph, pos_2d,
            nodelist=all_nodes,
            node_color='#E0E0E0',
            node_size=80,
            alpha=0.4
        )

        # Extract speeds from parsed instructions
        speed_map = {}
        landmark_idx = 0
        for seg in parsed_instruction.get("sub_routes", []):
            if seg.get("type") == "NAVIGATE":
                speed = seg.get("speed", "NORMAL")
                if landmark_idx < len(route) - 1:
                    speed_map[landmark_idx] = speed
                    landmark_idx += 1

        # Draw the complete path with shortest paths between landmark nodes
        full_path = []
        landmark_indices = []

        # For each consecutive pair of landmark nodes
        for i in range(len(route) - 1):
            # Compute shortest path between landmarks
            segment_path = self.du.get_shortest_path(route[i], route[i + 1])
            if segment_path:
                # Record the landmark position in the full path
                landmark_indices.append(len(full_path))
                # Add segment path to full path (excluding duplicates)
                if full_path and segment_path[0] == full_path[-1]:
                    full_path.extend(segment_path[1:])
                else:
                    full_path.extend(segment_path)

        # Make sure the last landmark is recorded
        if route and (not landmark_indices or landmark_indices[-1] != len(full_path) - 1):
            landmark_indices.append(len(full_path) - 1)

        # Draw the path segments with proper coloring
        if len(full_path) > 1:
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]

                # Determine which landmark segment this belongs to
                segment_idx = next((idx for idx, pos in enumerate(landmark_indices)
                                    if pos > i), len(landmark_indices)) - 1

                # Get the color based on speed
                speed = speed_map.get(segment_idx, "NORMAL")
                color = self.SPEED_COLORS.get(speed, self.SPEED_COLORS["NORMAL"])

                # Draw edge
                nx.draw_networkx_edges(
                    self.gl.graph, pos_2d,
                    edgelist=[(u, v)],
                    edge_color=color,
                    width=4.0
                )

        # Draw landmark nodes with highlight
        nx.draw_networkx_nodes(
            self.gl.graph, pos_2d,
            nodelist=route,
            node_color='#FF9800',
            node_size=200,
            alpha=1.0
        )

        # Add labels for route nodes
        labels = {n: str(n) for n in route}
        nx.draw_networkx_labels(
            self.gl.graph, pos_2d,
            labels=labels,
            font_size=12,
            font_weight='bold'
        )

        # Add speed legend
        legend_handles = []
        for speed, color in self.SPEED_COLORS.items():
            patch = plt.Line2D(
                [0], [0],
                color=color,
                lw=4,
                label=speed.replace('_', ' ').title()
            )
            legend_handles.append(patch)

        # Add the legend
        plt.legend(
            handles=legend_handles,
            loc='upper right',
            fontsize=12,
            title='Speed',
            framealpha=0.7
        )

        # Title and axis settings
        plt.title(title, fontsize=16, pad=15)
        plt.axis('off')
        plt.tight_layout()

        # Save figure if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Route visualization saved to {output_path}")

        # Show plot if requested
        if show_plot:
            plt.show()

        plt.close()