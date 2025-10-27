# author.ai

author.ai extracts atomic quantitative claims from raw prose and returns strict JSON via Pydantic models.

## Features
- Detects statistics, ratios (e.g. `1 in 5`), ranges (`10–12%`), and deltas (`up 2pp vs 2023`).
- Normalises numbers, percentage points, `per N` units, and locale-specific formats.
- Heuristically captures qualifiers, subjects, and nearby temporal references (years, quarters).
- Ships with a Click-powered CLI (`veritas-extract`) plus pytest-based regression tests and pre-commit hooks.

## Getting Started

### Prerequisites
- Python 3.11+
- [Poetry](https://python-poetry.org/)

### Installation
```bash
poetry install
```

### Run Tests
```bash
poetry run pytest
```

## CLI Usage
Read from stdin and pretty-print JSON:
```bash
echo "About 23.5% of UK A&E attendances in Q2 2024 breached the four-hour standard, up 2pp vs 2023." \
  | poetry run veritas-extract --pretty
```

Example output:
```json
[
  {
    "type": "statistic",
    "text": "About 23.5%",
    "span": {"start": 0, "end": 13},
    "quantity": 23.5,
    "unit": "%",
    "subject": "UK A&E attendances",
    "population": null,
    "time": "2024-Q2",
    "location": "UK",
    "qualifier": "about",
    "ratio": null,
    "range": null,
    "delta": null,
    "delta_direction": null,
    "baseline_time": null,
    "notes": []
  },
  {
    "type": "delta",
    "text": "up 2pp vs 2023",
    "span": {"start": 81, "end": 95},
    "quantity": null,
    "unit": "pp",
    "subject": null,
    "population": null,
    "time": null,
    "location": null,
    "qualifier": null,
    "ratio": null,
    "range": null,
    "delta": 2.0,
    "delta_direction": "up",
    "baseline_time": "2023",
    "notes": []
  }
]
```

Point the CLI at a file instead of stdin:
```bash
poetry run veritas-extract --infile data/sample.txt --pretty
```

## Tooling
- `poetry run ruff check .`
- `poetry run black .`
- `poetry run pre-commit run --all-files`
