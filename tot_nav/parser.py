"""
Command Parser Module for TOT-NAV
Parses natural-language navigation commands into structured JSON with:
- starting_node
- sub_routes (with type, from, to, direction, speed, confidence)
- sequence_confidence
"""
import os
import json
import logging
from typing import Any, Dict, List
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# System prompt with explicit direction in every segment, uncertainties removed
SYSTEM_PROMPT = """
You are a navigation instruction parser for a self-driving vehicle.
When given a user command, return exactly one JSON object with these keys:

- "starting_node": integer ID if user says "start at X", otherwise null.
- "sub_routes": an ordered array of route segments. Each segment must include:
  – "type": "NAVIGATE" or "TURN".
  – "from": origin of this segment (node ID or previous landmark).
  – "to": destination landmark for NAVIGATE, null for TURN.
  – "direction": one of "STRAIGHT", "LEFT", "RIGHT", "UTURN", or "NOT_SPECIFIED" for NAVIGATE when no explicit turn given.
  – "speed": one of VERY_SLOW, SLOW, NORMAL, FAST, VERY_FAST; for TURN segments, repeat last speed or default to NORMAL.
  – "confidence": float 0.0–1.0.
- "sequence_confidence": overall confidence 0.0–1.0.

Return only the JSON—no extra text.
"""


class CommandParser:
    def __init__(self):
        """
        Initialize the command parser with OpenAI client.
        Loads API key from environment variables.
        """
        # Load environment and initialize OpenAI client
        # Look for .env file in project root
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        load_dotenv(dotenv_path=env_path, override=True)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self.client = OpenAI(api_key=api_key)

    def parse(self, instruction: str) -> Dict[str, Any]:
        """
        Send the instruction to the LLM and parse the JSON response.

        Args:
            instruction: Natural language navigation instruction

        Returns:
            Structured dictionary with parsed navigation data
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction}
                ]
            )
            raw = response.choices[0].message.content.strip()
            # Strip code fences if present
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1])
            data = json.loads(raw)
            # Rename 'from' to 'from_' to avoid reserved keyword
            for seg in data.get("sub_routes", []):
                if "from" in seg:
                    seg["from_"] = seg.pop("from")
            return data
        except Exception as e:
            logger.error(f"Parsing error: {e}")
            raise

    def format_for_matching(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare landmark list for CLIP matching.

        Args:
            parsed: Parsed instruction data

        Returns:
            List of landmarks with confidence scores
        """
        landmarks = []
        for seg in parsed.get("sub_routes", []):
            if seg.get("type") == "NAVIGATE":
                landmarks.append({
                    "landmark": seg.get("to"),
                    "confidence": seg.get("confidence")
                })
        return landmarks

    def get_turn_segments(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract TURN segments for directional analysis.

        Args:
            parsed: Parsed instruction data

        Returns:
            List of turn segments with direction information
        """
        turns = []
        for seg in parsed.get("sub_routes", []):
            if seg.get("type") == "TURN":
                turns.append({
                    "origin": seg.get("from_"),
                    "direction": seg.get("direction"),
                    "confidence": seg.get("confidence")
                })
        return turns