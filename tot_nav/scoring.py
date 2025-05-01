"""
Scoring Module for TOT-NAV

Provides functions to:
- Calculate raw scores for route candidates based on heuristics
- Normalize scores across candidates
- Combine raw scores with LLM evaluations
"""

from typing import List, Dict, Any, Union


class ScoringFunction:
    def __init__(
            self,
            clip_weight: float = 2.0,
            parse_weight: float = 1.0,
            dist_weight: float = 1.5,
            turn_weight: float = 0.0,  # Set to 0 by default as LLM now handles turn evaluation
            llm_weight: float = 1.0
    ):
        """
        Initialize the scoring function with weighting parameters.

        Args:
            clip_weight: Weight for CLIP landmark matching confidence
            parse_weight: Weight for parser confidence
            dist_weight: Weight for normalized distance (negative impact)
            turn_weight: Weight for turn penalties (set to 0 if using LLM evaluation)
            llm_weight: Weight for LLM evaluation in final score
        """
        self.clip_w = clip_weight
        self.parse_w = parse_weight
        self.dist_w = dist_weight
        self.turn_w = turn_weight
        self.llm_w = llm_weight

    def raw_score(
            self,
            clip_conf: float,
            parse_conf: float,
            dist_norm: float,
            turn_penalty: float = 0.0
    ) -> float:
        """
        Calculate raw score for a route candidate.

        Args:
            clip_conf: CLIP confidence for landmark matching (0.0-1.0)
            parse_conf: Parser confidence (0.0-1.0)
            dist_norm: Normalized distance (usually distance/max_distance)
            turn_penalty: Optional penalty for turn violations (0.0-1.0)

        Returns:
            Raw score (higher is better)
        """
        return (
                self.clip_w * clip_conf +
                self.parse_w * parse_conf -
                self.dist_w * dist_norm -
                self.turn_w * turn_penalty
        )

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize a list of scores to range [0.0, 1.0].

        Args:
            scores: List of raw scores

        Returns:
            Normalized scores with same length as input
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # If all scores are identical, return 1.0 for all
        if max_score == min_score:
            return [1.0] * len(scores)

        # Scale to [0.0, 1.0] range
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def combine(
            self,
            raw_score: float,
            llm_score: float,
            alpha: float = 0.2
    ) -> float:
        """
        Combine raw score with LLM evaluation.

        Args:
            raw_score: Raw heuristic score
            llm_score: LLM evaluation score (0-10 range)
            alpha: Weight for raw score in combination (1-alpha for LLM)

        Returns:
            Combined score
        """
        # Normalize LLM score to 0-1 range
        normalized_llm = llm_score / 10.0

        # Combine scores with weighting
        return alpha * raw_score + (1 - alpha) * normalized_llm

    def score_candidates(
            self,
            candidates: List[Dict[str, Any]],
            llm_scores: Union[List[float], None] = None
    ) -> List[Dict[str, Any]]:
        """
        Score a list of route candidates.

        Args:
            candidates: List of candidate dictionaries with required fields
            llm_scores: Optional list of LLM scores (must match length of candidates)

        Returns:
            List of candidates with added 'score' field
        """
        # Extract raw scores
        raw_scores = [c.get('raw_score', 0.0) for c in candidates]

        # Normalize raw scores
        norm_scores = self.normalize_scores(raw_scores)

        # Combine with LLM scores if provided
        if llm_scores and len(llm_scores) == len(candidates):
            for i, (candidate, raw, norm, llm) in enumerate(
                    zip(candidates, raw_scores, norm_scores, llm_scores)
            ):
                candidate['raw_score'] = raw
                candidate['normalized_score'] = norm
                candidate['llm_score'] = llm
                candidate['combined_score'] = self.combine(norm, llm)
        else:
            # Use only raw scores
            for i, (candidate, raw, norm) in enumerate(
                    zip(candidates, raw_scores, norm_scores)
            ):
                candidate['raw_score'] = raw
                candidate['normalized_score'] = norm
                candidate['combined_score'] = norm

        return candidates