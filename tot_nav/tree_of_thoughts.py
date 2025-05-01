"""
Tree of Thoughts Module for TOT-NAV

Implements the core Tree of Thoughts algorithm using beam search to:
- Explore multiple navigation hypotheses simultaneously
- Evaluate and prune routes based on heuristic and LLM scoring
- Select optimal paths considering both landmark accuracy and directional constraints
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class NavigationState:
    """
    Represents a single navigation state in the search beam.
    Each state corresponds to a partial route with associated metrics.
    """

    def __init__(
            self,
            path: List[int],
            assigned_landmarks: int = 0,
            landmark: Optional[str] = None,
            distance: float = 0.0,
            raw_score: float = 0.0,
            llm_score: float = 0.0,
            combined_score: float = 0.0,
            segment_scores: Dict[int, float] = None
    ):
        """
        Initialize a navigation state.

        Args:
            path: Sequence of node IDs in the current path
            assigned_landmarks: Number of landmarks assigned so far
            landmark: Current landmark being navigated to
            distance: Total path distance
            raw_score: Raw heuristic score
            llm_score: Cumulative LLM evaluation score
            combined_score: Combined score for ranking
            segment_scores: Dictionary of individual segment scores
        """
        self.path = path
        self.assigned_landmarks = assigned_landmarks
        self.landmark = landmark
        self.distance = distance
        self.raw_score = raw_score
        self.llm_score = llm_score
        self.combined_score = combined_score
        self.segment_scores = segment_scores or {}

    def __repr__(self) -> str:
        """String representation of the state for debugging."""
        return (
            f"NavigationState(path={self.path}, "
            f"assigned={self.assigned_landmarks}, "
            f"landmark='{self.landmark}', "
            f"distance={self.distance:.2f}, "
            f"raw={self.raw_score:.2f}, "
            f"llm={self.llm_score:.2f}, "
            f"combined={self.combined_score:.2f})"
        )

    def extend(self, node_id: int, landmark: str, distance_delta: float) -> 'NavigationState':
        """
        Create a new state by extending the current path with a new node.

        Args:
            node_id: Node ID to add to the path
            landmark: Landmark associated with the new node
            distance_delta: Additional distance to the new node

        Returns:
            A new NavigationState object with the extended path
        """
        new_path = self.path + [node_id]
        new_distance = self.distance + distance_delta

        return NavigationState(
            path=new_path,
            assigned_landmarks=self.assigned_landmarks + 1,
            landmark=landmark,
            distance=new_distance,
            raw_score=self.raw_score,
            llm_score=self.llm_score,
            combined_score=self.combined_score,
            segment_scores=self.segment_scores.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation for LLM evaluation."""
        return {
            "path": self.path,
            "assigned": self.assigned_landmarks,
            "landmark": self.landmark,
            "distance": self.distance,
            "raw_score": self.raw_score,
            "llm_score": self.llm_score,
            "combined_score": self.combined_score,
            "segment_scores": self.segment_scores
        }


class TreeOfThoughts:
    """
    Implements the Tree of Thoughts algorithm for navigation planning.
    """

    def __init__(
            self,
            beam_width: int = 5,
            k_filter: int = 15
    ):
        """
        Initialize the Tree of Thoughts algorithm.

        Args:
            beam_width: Maximum number of states to keep in beam
            k_filter: Number of candidates to evaluate with LLM
        """
        self.beam_width = beam_width
        self.k_filter = k_filter

    def initialize_beam(
            self,
            starting_node: Optional[int] = None,
            default_node: int = 0
    ) -> List[NavigationState]:
        """
        Create initial beam with starting state.

        Args:
            starting_node: Starting node ID if specified in instruction
            default_node: Default starting node if not specified

        Returns:
            List containing the initial state
        """
        start = starting_node if starting_node is not None else default_node
        initial_state = NavigationState(
            path=[start],
            assigned_landmarks=0,
            landmark=None,
            distance=0.0,
            raw_score=0.0,
            llm_score=0.0,
            combined_score=0.0
        )
        return [initial_state]

    def expand_states(
            self,
            beam: List[NavigationState],
            landmark: str,
            landmark_matches: List[Dict[str, Any]],
            get_distance_fn: Callable[[int, int], float],
            scoring_fn: Callable[[float, float, float], float],
            landmark_confidence: float = 0.8,
            parse_confidence: float = 0.8
    ) -> List[NavigationState]:
        """
        Expand each state in the beam with potential matches for the current landmark.

        Args:
            beam: Current beam of states
            landmark: Current landmark description
            landmark_matches: List of potential node matches for the landmark
            get_distance_fn: Function to get distance between nodes
            scoring_fn: Function to compute raw score for a candidate
            landmark_confidence: Confidence in the landmark parsing
            parse_confidence: Overall parsing confidence

        Returns:
            List of new candidate states
        """
        candidates = []

        for state in beam:
            current_node = state.path[-1]

            for match in landmark_matches:
                node_id = match["node_id"]
                confidence = match["confidence"]

                # Skip if already in path
                if node_id in state.path:
                    continue

                # Get distance to this node
                distance = get_distance_fn(current_node, node_id)

                # Create new state
                new_state = state.extend(node_id, landmark, distance)

                # Compute normalized distance (simplified version)
                dist_norm = distance / 1000.0

                # Calculate raw score for this extension
                raw_score_delta = scoring_fn(
                    confidence,
                    parse_confidence,
                    dist_norm
                )

                # Update scores
                new_state.raw_score = state.raw_score + raw_score_delta

                candidates.append(new_state)

        return candidates

    def filter_candidates(
            self,
            candidates: List[NavigationState],
            by_score: str = "raw_score"
    ) -> List[NavigationState]:
        """
        Filter candidates to keep only the top-k by specified score.

        Args:
            candidates: List of candidate states
            by_score: Score field to use for filtering

        Returns:
            Filtered list of candidates
        """
        if not candidates:
            return []

        # Sort by specified score
        candidates.sort(key=lambda x: getattr(x, by_score, 0.0), reverse=True)

        # Take top-k candidates
        return candidates[:min(self.k_filter, len(candidates))]

    def update_with_llm_scores(
            self,
            candidates: List[NavigationState],
            llm_scores: List[float],
            segment_idx: int,
            combine_fn: Callable[[float, float], float]
    ) -> List[NavigationState]:
        """
        Update candidates with LLM evaluation scores.

        Args:
            candidates: List of candidate states
            llm_scores: List of LLM scores (same length as candidates)
            segment_idx: Current segment index
            combine_fn: Function to combine raw and LLM scores

        Returns:
            Updated list of candidates
        """
        for i, (candidate, score) in enumerate(zip(candidates, llm_scores)):
            # Save segment score
            candidate.segment_scores[segment_idx] = score

            # Update cumulative LLM score
            candidate.llm_score = candidate.llm_score + score

            # Recalculate combined score
            candidate.combined_score = combine_fn(candidate.raw_score, candidate.llm_score)

        return candidates

    def prune_beam(
            self,
            candidates: List[NavigationState],
            by_score: str = "combined_score"
    ) -> List[NavigationState]:
        """
        Prune candidates to beam width.

        Args:
            candidates: List of candidate states
            by_score: Score field to use for pruning

        Returns:
            Pruned beam of states
        """
        if not candidates:
            return []

        # Sort by specified score
        candidates.sort(key=lambda x: getattr(x, by_score, 0.0), reverse=True)

        # Take top beam_width candidates
        return candidates[:min(self.beam_width, len(candidates))]

    def select_final_route(
            self,
            complete_states: List[NavigationState],
            incomplete_states: List[NavigationState],
            user_resolver: Optional[Callable[[List[NavigationState]], NavigationState]] = None
    ) -> NavigationState:
        """
        Select final route from complete states, falling back to incomplete if needed.

        Args:
            complete_states: States that have assigned all landmarks
            incomplete_states: States that are still partial
            user_resolver: Optional callback for user to resolve ties

        Returns:
            Selected final state
        """
        # Prefer complete states if available
        final_states = complete_states if complete_states else incomplete_states

        if not final_states:
            raise ValueError("No valid routes found")

        # Sort by LLM score
        final_states.sort(key=lambda x: x.llm_score, reverse=True)

        # Get the best score
        best_score = final_states[0].llm_score

        # Find all states with the best score
        best_states = [s for s in final_states if s.llm_score == best_score]

        # If multiple best states and user resolver provided, use it
        if len(best_states) > 1 and user_resolver:
            return user_resolver(best_states)

        # Otherwise return the first best state
        return best_states[0]