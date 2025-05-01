"""
TOT-NAV Pipeline Module

Main entry point for the TOT-NAV system that integrates all components:
- Command parsing
- Landmark matching
- Tree of Thoughts navigation planning
- LLM evaluation
- User interaction for uncertainty resolution
"""

import logging
import os
from typing import List, Dict, Any, Optional, Callable, Tuple

from tot_nav.parser import CommandParser
from tot_nav.matching import CLIPMatcher
from tot_nav.graph_loader import GraphLoader
from tot_nav.utils import DistanceUtils, Turn
from tot_nav.turns import TurnAnalyzer
from tot_nav.scoring import ScoringFunction
from tot_nav.tree_of_thoughts import TreeOfThoughts, NavigationState
from tot_nav.llm import LLMClient
from tot_nav.visualizer import RouteVisualizer
import os


logger = logging.getLogger(__name__)


class TOTNavPipeline:
    def __init__(
            self,
            graph_pkl: str,
            beam_width: int = 5,
            k_filter: int = 15,
            clip_candidates: int = 5,
            output_dir: str = "output"
    ):
        # Existing initialization code
        self.gl = GraphLoader(graph_pkl)
        self.du = DistanceUtils(self.gl.graph)
        self.parser = CommandParser()
        self.matcher = CLIPMatcher(self.gl, clip_candidates)
        self.turn_analyzer = TurnAnalyzer(self.gl, self.du)
        self.scoring = ScoringFunction(
            clip_weight=2.0,
            parse_weight=1.0,
            dist_weight=1.5
        )
        self.tot = TreeOfThoughts(
            beam_width=beam_width,
            k_filter=k_filter
        )
        self.llm = LLMClient()

        # Add visualizer
        self.visualizer = RouteVisualizer(self.gl, self.du)

        # Configuration
        self.beam_width = beam_width
        self.k_filter = k_filter
        self.clip_candidates = clip_candidates
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"Initialized TOT-NAV pipeline with beam_width={beam_width}, "
            f"k_filter={k_filter}, clip_candidates={clip_candidates}"
        )

    def run(self, instruction: str, visualize: bool = True) -> List[int]:
        """
        Execute the full TOT-NAV pipeline on a natural language instruction.

        Args:
            instruction: Natural language navigation instruction
            visualize: Whether to generate a visualization of the route

        Returns:
            List of node IDs forming the optimal navigation path
        """
        # 1. Parse instruction
        logger.info(f"Parsing instruction: {instruction}")
        parsed = self.parser.parse(instruction)
        logger.info(f"Parsed instruction: {parsed}")

        # Get starting node
        starting_node = parsed.get("starting_node", None)

        # Get sequence confidence
        sequence_confidence = parsed.get("sequence_confidence", 0.8)

        # 2. Extract landmarks
        landmarks = []
        for seg in parsed.get("sub_routes", []):
            if seg.get("type") == "NAVIGATE":
                landmarks.append(seg.get("to"))
        logger.info(f"Extracted landmarks: {landmarks}")

        # 3. Match landmarks to nodes
        landmark_matches = {}
        for landmark in landmarks:
            matches = self.matcher.match([landmark])
            landmark_matches[landmark] = matches.get(landmark, [])
        logger.info(f"Landmark matches: {landmark_matches}")

        # 4. Initialize beam
        beam = self.tot.initialize_beam(starting_node)
        complete_states = []

        # 5. For each landmark in order
        for i, landmark in enumerate(landmarks):
            logger.info(f"Processing landmark {i + 1}/{len(landmarks)}: {landmark}")

            # Skip if no matches
            if not landmark_matches[landmark]:
                logger.warning(f"No matches for landmark '{landmark}'")
                continue

            # Get segment confidence
            segment_confidence = 0.8  # Default
            for seg in parsed.get("sub_routes", []):
                if seg.get("type") == "NAVIGATE" and seg.get("to") == landmark:
                    segment_confidence = seg.get("confidence", 0.8)
                    break

            # a. Expand beam with potential matches
            candidates = self.tot.expand_states(
                beam=beam,
                landmark=landmark,
                landmark_matches=landmark_matches[landmark],
                get_distance_fn=lambda a, b: self._get_path_distance(a, b),
                scoring_fn=lambda clip_conf, parse_conf, dist_norm: self.scoring.raw_score(
                    clip_conf=clip_conf,
                    parse_conf=parse_conf,
                    dist_norm=dist_norm
                ),
                landmark_confidence=segment_confidence,
                parse_confidence=sequence_confidence
            )

            if not candidates:
                logger.warning("No valid candidates generated")
                break

            # b. Filter candidates for LLM evaluation
            filtered_candidates = self.tot.filter_candidates(candidates)

            # c. Evaluate with LLM
            llm_scores = self._evaluate_candidates_with_llm(
                candidates=filtered_candidates,
                landmark=landmark,
                segment_idx=i
            )

            # d. Update candidates with LLM scores
            updated_candidates = self.tot.update_with_llm_scores(
                candidates=filtered_candidates,
                llm_scores=llm_scores,
                segment_idx=i,
                combine_fn=lambda raw, llm: self.scoring.combine(
                    raw_score=raw,
                    llm_score=llm,
                    alpha=0.2  # 20% raw, 80% LLM
                )
            )

            # e. Prune beam
            beam = self.tot.prune_beam(updated_candidates)

            # f. Collect completed states
            complete = [s for s in beam if s.assigned_landmarks == len(landmarks)]
            complete_states.extend(complete)
            beam = [s for s in beam if s.assigned_landmarks < len(landmarks)]

        try:
            final_state = self.tot.select_final_route(
                complete_states=complete_states,
                incomplete_states=beam,
                user_resolver=self._ask_user_to_break_tie
            )
            logger.info(f"Selected route: {final_state.path}")

            # Visualize the route if requested
            if visualize and final_state.path:
                # Generate a filename based on instruction
                safe_filename = "".join(x for x in instruction[:30] if x.isalnum() or x.isspace()).strip().replace(" ",
                                                                                                                   "_")
                if not safe_filename:
                    safe_filename = "route"

                output_path = os.path.join(self.output_dir, f"{safe_filename}.png")

                # Create visualization
                self.visualizer.visualize_route(
                    route=final_state.path,
                    parsed_instruction=parsed,
                    output_path=output_path,
                    title=f"TOT-NAV Route: {instruction[:50]}{'...' if len(instruction) > 50 else ''}",
                    show_plot=False  # Don't block execution with plot window
                )
                logger.info(f"Route visualization saved to {output_path}")

            return final_state.path

        except ValueError as e:
            logger.error(f"Route selection failed: {e}")
            if starting_node is not None:
                return [starting_node]
            return [0]  # Fallback to node 0
    def _get_path_distance(self, start_node: int, end_node: int) -> float:
        """
        Get the distance between two nodes.

        Args:
            start_node: Starting node ID
            end_node: Ending node ID

        Returns:
            Distance between nodes
        """
        path = self.du.get_shortest_path(start_node, end_node)
        if path:
            return self.du.get_path_distance(path)
        return float('inf')

    def _evaluate_candidates_with_llm(
            self,
            candidates: List[NavigationState],
            landmark: str,
            segment_idx: int
    ) -> List[float]:
        """
        Evaluate candidates using LLM.

        Args:
            candidates: List of candidate states
            landmark: Current landmark being evaluated
            segment_idx: Index of current segment

        Returns:
            List of LLM scores for candidates
        """
        scores = []

        for i, candidate in enumerate(candidates):
            # Get path info
            path_info = self.turn_analyzer.get_path_info(candidate.path)

            # Build candidate summary
            candidate_summary = self.turn_analyzer.summarize_candidate(
                path=candidate.path,
                landmark=landmark
            )

            # Build prompt for LLM
            prompt = self._build_evaluation_prompt(
                candidate=candidate_summary,
                segment_idx=segment_idx,
                landmark=landmark,
                alternatives=candidates[:i] + candidates[i + 1:]
            )

            # Get LLM score
            score = self.llm.evaluate_single(prompt)
            scores.append(score)

        return scores

    def _build_evaluation_prompt(
            self,
            candidate: Dict[str, Any],
            segment_idx: int,
            landmark: str,
            alternatives: List[NavigationState]
    ) -> str:
        """
        Build prompt for LLM evaluation.

        Args:
            candidate: Candidate summary
            segment_idx: Current segment index
            landmark: Current landmark
            alternatives: Other candidates for context

        Returns:
            Formatted prompt for LLM
        """
        lines: List[str] = []

        # Get relevant segment information from the parsed instruction we stored during the run
        seg = None
        # We need to get this from the parsed instruction that was already passed to run()
        parsed_instruction = self.parser.parse(self.instruction) if hasattr(self, 'instruction') else {}

        for s in parsed_instruction.get("sub_routes", []):
            if s.get("type") == "NAVIGATE" and s.get("to") == landmark:
                seg = s
                break

        if seg is None:
            # Create a default segment for fallback
            seg = {"type": "NAVIGATE", "to": landmark, "direction": "NOT_SPECIFIED"}

        # 1) Header with candidate numbering
        full_instr = getattr(self, 'instruction', f"Navigate to {landmark}")
        total_candidates = len(alternatives) + 1
        candidate_idx = 0  # Current candidate is always the first

        lines.append(f"Hi, consider you are a robot navigation assistant evaluating candidate "
            f"{candidate_idx+1}/{total_candidates} for the instruction:\n"
            f"  “{full_instr}”\n"
        )

        # 2) Which segment are we satisfying?
        lines.append("You currently are evaluating this segment:")
        if seg['type'] == 'NAVIGATE':
            lines.append(f"  • Go to “{seg['to']}” direction {seg['direction']}\n")
        else:
            lines.append(f"  • Perform a {seg.get('direction', 'NOT_SPECIFIED')} TURN\n")

        # 3) Summarize how we got here
        at_node = candidate.get('at_node', 0)
        landmark_text = candidate.get('landmark', '<start>')
        lines.append(f"Your current position: Node {at_node} {landmark_text}\n")

        lines.append("Turn-by-turn navigation so far:")
        turns_str = candidate.get('turns_str', '')
        for step in turns_str.split(", ") if turns_str else []:
            lines.append(f"   • {step}")
        lines.append(f"\nTotal turns so far: {len(turns_str.split(', ')) if turns_str else 0}\n")

        lines.append(f"Actual route so far (full path): {candidate.get('path_str', '')}")
        lines.append(f"Distance so far: {candidate.get('distance', 0.0):.1f} meters")
        lines.append(f"Turns so far: {turns_str}\n")

        # 4) What moves are available now?
        avail = candidate.get('available_turns', {})
        if avail:
            avail_list = ", ".join(avail.keys())
            lines.append(f"Your available turns from here: {avail_list}\n")
        else:
            lines.append("No turns data available at current node.\n")

        # 5) Guided evaluation criteria
        lines.append("Please evaluate this candidate on:")
        lines.append("  1) Fidelity to the requested direction(s).")
        lines.append("  2) Appropriateness of current position to continue the segment.")
        lines.append("  3) Avoidance of undesired or missing turns.")
        lines.append(
            "  4) Does it perform better or worse than the rest? If it's the most promising compared to rest you can go high (7-10), "
            "if it's clearly not a good candidate also can be harsh with low low numbers (0-5), if it could be but has some flaws (5-7).\n"
        )
        # Add information about previous segment scores if available
        if hasattr(alternatives[0], 'segment_scores'):
            segment_scores = []
            state = alternatives[0].to_dict() if alternatives else {}
            for i in range(state.get('assigned', 0)):
                segment_key = f"segment_{i}_llm_score"
                if segment_key in state:
                    segment_scores.append(f"Segment {i}: {state[segment_key]:.1f}")

            if segment_scores:
                lines.append(
                    "Previous segment ratings: CONSIDER THEM TO EVALUATE IF ITS WORTH KEEP RATING THIS PARTIAL ROUTE:")
                for score_info in segment_scores:
                    lines.append(f"  • {score_info}")
                lines.append("")  # blank line

        # 6) Detailed summary of alternatives for context
        if alternatives:
            lines.append("ALTERNATIVE ROUTES (full path & turns):")
            for i, alt_state in enumerate(alternatives, start=1):
                alt = self.turn_analyzer.summarize_candidate(alt_state.path, landmark)
                lines.append(f"Route #{i}:")
                lines.append(f"  • Full path: {alt['path_str']}")
                lines.append(f"  • Total distance: {alt['distance']:.1f} meters")
                avail_moves = alt.get('available_turns', {}).keys()
                lines.append(f"  • Available moves at goal node: {', '.join(avail_moves)}")
                lines.append("")  # blank line

        # 7) The actual ask
        lines.append(
            "Rate *this* candidate from 0.0 (worst) to 10.0 (best). You have decimals available so use them, even two, it's important to be really specific... "
            "USE THE DECIMALS IMAGINING IS 0-100, I need you to be specific because there might be ties and i don't want that."
        )
        lines.append("Return a single numeric score only.")

        return "\n".join(lines)

    def _ask_user_to_break_tie(self, tied_states: List[NavigationState]) -> NavigationState:
        """
        Ask user to choose between tied routes.

        Args:
            tied_states: List of states with tied scores

        Returns:
            User-selected state
        """
        print("\nMultiple equally good routes found. Please choose one:")

        for i, state in enumerate(tied_states):
            # Format information about the state
            path_str = " → ".join(str(n) for n in state.path)
            segments_str = ", ".join(
                f"{i + 1}: {score:.1f}" for i, score in state.segment_scores.items()
            )

            print(f"{i + 1}: Path = {path_str}")
            print(f"   Distance = {state.distance:.1f}m, LLM Score = {state.llm_score:.1f}")
            print(f"   Segment Scores = [{segments_str}]")
            print()

        # Get user choice
        while True:
            try:
                choice = int(input(f"Enter your choice (1-{len(tied_states)}): "))
                if 1 <= choice <= len(tied_states):
                    return tied_states[choice - 1]
                print(f"Please enter a number between 1 and {len(tied_states)}")
            except ValueError:
                print("Please enter a valid number")