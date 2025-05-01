"""
LLM Client Module for TOT-NAV

Provides an interface to Large Language Models for:
- Route evaluation and scoring
- Prompt construction and response parsing
- Score extraction and normalization
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with LLMs to evaluate navigation routes.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the LLM client.

        Args:
            model: LLM model to use (default: gpt-4o)
        """
        # Load API key from environment
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        load_dotenv(dotenv_path=env_path, override=True)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model

        logger.info(f"Initialized LLM client with model: {model}")

    def _strip_fences(self, text: str) -> str:
        """
        Remove code fences from text if present.

        Args:
            text: Text that might contain code fences

        Returns:
            Cleaned text
        """
        if text.startswith("```"):
            return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        return text.strip()

    def evaluate_single(self, prompt: str) -> float:
        """
        Evaluate a single candidate route and return a score from 1-10.

        Args:
            prompt: Evaluation prompt for the LLM

        Returns:
            Score between 1.0 and 10.0
        """
        try:
            logger.debug(f"LLM evaluation prompt:\n{prompt}")

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are evaluating navigation routes. Provide detailed reasoning followed by a score from 1-10."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )

            text = resp.choices[0].message.content.strip()
            logger.info(f"LLM evaluation response: {text}")

            # Extract the score using regex - looking for SCORE: X.X pattern
            score_pattern = re.compile(r'SCORE:\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
            match = score_pattern.search(text)

            if match:
                score = float(match.group(1))
                return min(10.0, max(1.0, score))  # Clamp to 1-10 range

            # Fallback: look for any number in the text
            number_pattern = re.compile(r'(\d+(?:\.\d+)?)')
            matches = number_pattern.findall(text)

            if matches:
                # Try to find numbers that look like scores (between 1 and 10)
                possible_scores = [float(m) for m in matches if 1 <= float(m) <= 10]
                if possible_scores:
                    return possible_scores[-1]  # Take the last number in the valid range

                # If no valid scores found, use the last number and normalize it
                logger.warning(f"No valid score found in range 1-10, using fallback: {matches[-1]}")
                raw_score = float(matches[-1])

                # Normalize score to 1-10 range
                if raw_score > 10:
                    return min(10.0, raw_score / 10)
                return min(10.0, max(1.0, raw_score))

            logger.error(f"Could not extract score from LLM response: {text}")
            return 5.0  # Default middle score

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return 5.0  # Default middle score

    def evaluate_batch(self, prompts: List[str]) -> List[float]:
        """
        Evaluate multiple candidate routes in sequence.

        Args:
            prompts: List of evaluation prompts

        Returns:
            List of scores between 1.0 and 10.0
        """
        return [self.evaluate_single(prompt) for prompt in prompts]
