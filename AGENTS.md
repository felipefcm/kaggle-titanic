# Agent Guidelines for Titanic Project

This document provides guidelines for agentic coding agents operating in this repository.

## Project Overview

A Kaggle Titanic competition submission using PyTorch neural networks for passenger survival prediction. Python 3.14+ project using Poetry for dependency management.

## Build, Lint & Test Commands

### Environment Setup
```bash
poetry install          # Install dependencies
poetry lock            # Update poetry.lock
```

### Running Code
```bash
poetry run python -m titanic.check_missing           # Check missing values in dataset
poetry run python -m titanic.nn.train                # Train the neural network model
```

### Testing
```bash
poetry run pytest tests/                             # Run all tests
poetry run pytest tests/test_file.py                 # Run single test file
poetry run pytest tests/test_file.py::test_func      # Run single test function
poetry run pytest -v                                 # Verbose test output
poetry run pytest --tb=short                         # Short traceback format
```

### Code Quality
Currently, no linting tools (pylint, flake8, black) are configured. Follow the style guidelines below.

## Code Style Guidelines

### Imports
- Use `from X import Y` for specific items, `import X` for modules
- Example:
  ```python
  import pandas as pd
  from dataclasses import dataclass
  from typing import Optional, Tuple, cast, List
  
  from torch import Tensor
  import torch.nn as nn
  
  from titanic.dataset import Passenger
  ```

### Formatting
- Line length: Aim for ≤100 characters (no hard limit enforced)
- Indentation: 4 spaces (Python standard)
- Use type hints on function signatures and dataclass fields
- Single blank line between methods/functions, two between class definitions

### Types & Type Hints
- Always annotate function parameters and return types
- Use `list[X]` or `List[X]` for list types
- Use `Optional[X]` for nullable values (not bare `X | None`)
- Use `cast()` from typing module when converting values with uncertain types
- Use `Tuple[X, Y]` from typing for return tuples
- Example:
  ```python
  def load_data(file_path: str) -> Tuple[list[Passenger], Stats]:
      age: Optional[float] = None
  ```

### Dataclasses & Data Structures
- Use `@dataclass` decorator for data containers
- Use type hints on all fields
- Default values are acceptable for optional fields
- Example:
  ```python
  @dataclass
  class Passenger:
      id: int = 0
      survived: bool = False
      age: Optional[float] = None
  ```

### Key Project Modules
- `titanic.dataset`: Data loading (Passenger dataclass, load_titanic_data)
- `titanic.nn.input`: Feature engineering for torch tensors
- `titanic.nn.nn`: Neural network model definition
- `titanic.nn.train`: Training loop and hyperparameters
- `titanic.check_missing`: Dataset analysis utilities

## Project Dependencies
- **Data**: pandas, numpy
- **ML**: torch, scikit-learn, tensorboard
- **Jupyter**: ipykernel, matplotlib

### Documentation

Use Context7 MCP server to obtain documentation for libraries.
