# OM-501 Spring 2024

Python utilities package for the OM-501 optimization course.

## Overview

Reusable utilities for solving optimization problems including the Traveling Salesman Problem and local search heuristics.

## Modules

| Module | Description |
|--------|-------------|
| `optutilities/tsp.py` | TSP data generation and visualization |
| `optutilities/improvement.py` | Local search operators (pairwise interchange, subsequence reversal, three-opt) |

## Usage

```python
from optutilities import tsp, improvement

# Generate random TSP instance
data = tsp.generate_random_tsp_data(n_cities=20)

# Apply improvement heuristics
improved_tour = improvement.adjacent_pairwise_interchange(tour)
```

## Dependencies

- gurobipy
- pandas
- numpy
- scikit-learn
- matplotlib
