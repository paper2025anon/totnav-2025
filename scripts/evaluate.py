#!/usr/bin/env python3
"""
Evaluate TOT-NAV on a given navigation instruction.
Usage: python evaluate.py "Look for a blue SUV, then go straight to a white house"
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import tot_nav
sys.path.append(str(Path(__file__).parent.parent))

from tot_nav.pipeline import TOTNavPipeline


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate TOT-NAV on a navigation instruction")
    parser.add_argument(
        "instruction",
        help="Natural language navigation instruction"
    )
    parser.add_argument(
        "--graph",
        default="data/graph_large.pkl",
        help="Path to graph pickle file"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width for Tree of Thoughts"
    )
    parser.add_argument(
        "--k-filter",
        type=int,
        default=15,
        help="K filter for LLM evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable route visualization"
    )

    args = parser.parse_args()

    # Initialize TOT-NAV pipeline
    pipeline = TOTNavPipeline(
        graph_pkl=args.graph,
        beam_width=args.beam_width,
        k_filter=args.k_filter,
        output_dir=args.output_dir
    )

    # Run navigation
    print(f"\nNavigating with instruction: \"{args.instruction}\"")
    print("=" * 80)

    route = pipeline.run(
        instruction=args.instruction,
        visualize=not args.no_visualization
    )

    # Output results
    print("\n" + "=" * 80)
    print("🏁 Navigation complete!")
    print(f"Final route: {' → '.join(str(node) for node in route)}")

    if not args.no_visualization:
        print(f"Visualization saved to {args.output_dir}/")


if __name__ == "__main__":
    main()