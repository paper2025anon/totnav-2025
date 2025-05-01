# TOT-NAV: Uncertainty-aware, Context-driven Robot Navigation with a Tree of Thoughts

## Overview

TOT-NAV is an embodied navigation system designed to overcome limitations in current robot navigation approaches. Unlike methods that treat navigation as a landmark-only task, TOT-NAV integrates semantic information such as direction and speed, and provides robust mechanisms for managing uncertainty in both user instructions and model predictions.

## Core Concept: Tree of Thoughts for Navigation

Instead of relying on a single path, TOT-NAV explores multiple navigation hypotheses simultaneously through a Tree of Thoughts (ToT) approach:

1. **Parse** complex natural language instructions into structured semantic data
2. **Expand** multiple candidate paths for each landmark
3. **Filter** candidates using a fast heuristic scoring
4. **Evaluate** promising candidates with an LLM for contextual understanding
5. **Select** optimal paths considering landmarks, turns, and user constraints
6. **Engage** users only when uncertainty remains high

This approach enables backtracking, re-planning, and uncertainty management without requiring additional supervised data.

## Key Components

- **CommandParser**: Transforms free-form navigation instructions into structured data
- **ClipMatcher**: Matches textual landmark descriptions to visual observations
- **TurnAnalyzer**: Computes path geometry and enforces directional constraints
- **ToT Planner**: Maintains multiple navigation hypotheses using beam search
- **LLM Evaluator**: Provides contextual scoring based on route characteristics
- **User Interaction**: Engages users to resolve ambiguities when confidence is low

## Installation

```bash
# Clone the repository
git clone https://github.com/paper2025anon/totnav-2025.git
cd ToT-Nav

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Edit/Create .env with your API keys for OpenAI
```

## Usage

```

# Initialize ToT-Nav with environment graph
navigator = ToTNav(
    graph_pkl="data/graph_small.pkl",
    beam_width=5,
    k_filter=15
)

# Run navigation from natural language instruction
instruction = "Look for a blue SUV, then go straight to a white house, take a U-turn slowly, and go straight until you find a white truck."
route = navigator.run(instruction)

# Display the resulting path
print(f"Route: {' → '.join(str(node) for node in route)}")
```

## Repository Structure

```
ToT-Nav/
├── README.md               # Project documentation
├── data/                   # Sample navigation environments
├── requirements.txt        # Project dependencies
├── scripts/                # Utility scripts
│   ├── evaluate.py         # Evaluation metrics computation
└── tot_nav/                # Core package
    ├── __init__.py
    ├── parser.py           # Instruction parsing module
    ├── matching.py         # Landmark matching with CLIP
    ├── turns.py            # Turn analysis and directional constraints
    ├── scoring.py          # Scoring functions and metrics
    ├── pipeline.py         # Workflow implementation
    ├── llm.py              # LLM integration for evaluation
    └── tree_of_thoughts.py # Main Tree of Thoughts implementation
```

## Results

TOT-NAV significantly outperforms existing methods across various metrics:
- 85% planning success rate (vs. 65% baseline)
- 94.7% planning efficiency 
- 85.8% contextual granularity (instruction fidelity)

Our ablation studies demonstrate the importance of each component, with the full system achieving substantially better results than partial implementations.

## Citation

If you use TOT-NAV in your research, please cite our paper:

```bibtex
@inproceedings{anonymous2025totnav,
  title={TOT-NAV: Uncertainty-aware, Context-driven Robot Navigation via a Tree of Thoughts},
  author={Anonymous},
  booktitle={Conference on Robot Learning},
  year={2025}
}
```

## License.
