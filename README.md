# SE3062 Practical 06 – Genetic Algorithms for 0/1 Knapsack

## Project Purpose
This project implements a Genetic Algorithm (GA) framework using Python and DEAP (`Distributed Evolutionary Algorithms in Python`) to solve the classic NP-hard **0/1 Knapsack Problem**. The implementation conducts a rigorous, empirical study analyzing how different genetic operators (selection pressure, crossover mechanics, mutation rates, and elitism) affect convergence speed, constraint satisfaction, and solution optimality under a quadratic-order hard-penalty function.

All algorithms, experiments, benchmark parsers, and plotting routines are self-contained within [`ga_knapsack.py`](file:///c:/CS/Y3S1/IS/labs/lab6/SE3062-Practical06/ga_knapsack.py).

---

## Benchmark Dataset
- **Benchmark Source**: David Pisinger's Combinatorial Optimization Benchmarks  
  Benchmark URL: [https://hjemmesider.diku.dk/~pisinger/codes.html](https://hjemmesider.diku.dk/~pisinger/codes.html)  
  *(Archive: `smallcoeff_pisinger.tgz`)*
- **Selected File**: `knapPI_1_100_1000.csv`
- **Instance Identifier**: `knapPI_1_100_1000_1`
  - Number of items ($n$): `100`
  - Knapsack capacity ($C$): `995`
  - Known global optimum ($z$): `9147` (proven optimal knapsack value with optimal weight $W = 985 \le 995$)
  - Hard penalty multiplier: $M = 10 \times \max(values) = 10 \times 997 = 9970$

*Note: The dataset was authored by David Pisinger. It is used here solely as a standard standardized academic benchmark. The global optimum $z=9147$ is used strictly as a post-run evaluation reference and is never provided to or used within the GA.*

---

## Environment & Dependencies

- **Python Version**: Python 3.10+ (tested on Python 3.11 / 3.12)
- **Required Libraries**:
  - `deap` (Evolutionary computation framework)
  - `numpy` (Numerical vector operations and array manipulation)
  - `pandas` (Structured tabular results collection and CSV export)
  - `matplotlib` (Publication-quality plotting and visualization)

### Virtual Environment Setup & Installation

```powershell
# 1. Create a virtual environment (if not already created)
python -m venv .venv

# 2. Activate the virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# 3. Install required packages
pip install deap numpy pandas matplotlib
```

---

## Baseline Parameters

| Parameter | Baseline Value | Description |
| :--- | :--- | :--- |
| **Representation** | Binary List ($n=100$) | $1 = \text{item selected}$, $0 = \text{item not selected}$ |
| **Population Size ($N$)** | `150` | Number of chromosomes per generation |
| **Generations ($G$)** | `200` | Total generational iterations |
| **Selection** | Tournament ($k=3$) | Tournament selection with size 3 |
| **Crossover Operator** | 2-Point (`tools.cxTwoPoint`) | Two cut points swapped between parent pairs |
| **Crossover Probability ($p_c$)**| `0.9` | 90% chance of crossover per offspring pair |
| **Mutation Operator** | Bit-Flip (`tools.mutFlipBit`) | Independent bit-flip per gene |
| **Mutation Probability ($p_m$)** | `0.02` | 2% probability of flipping each bit |
| **Elitism ($E$)** | `2` | Top 2 individuals copied unchanged to next generation |
| **Fitness Formulation** | Hard Penalty | $\text{Fitness}(x) = \sum v_i x_i - M \cdot \max(0, \sum w_i x_i - C)$ |
| **Random Seeds** | `[42, 43, 44, 45, 46]` | 5 independent reproducible seeds per configuration |

---

## Running the Experiments

To run all 50 experiments (10 configurations $\times$ 5 seeds), generate all summary metrics, and produce the comparison plots, simply run:

```powershell
python ga_knapsack.py
```

The script will automatically:
1. Extract `knapPI_1_100_1000.csv` from `smallcoeff_pisinger.tgz` into `data/pisinger/` if not present.
2. Validate the 100 items, capacity $C=995$, and known optimum $z=9147$.
3. Execute all 10 configurations across the 5 random seeds with real elitism.
4. Calculate convergence metrics relative to the baseline median final best fitness.
5. Export `results/experiment_results.csv` (run-level data) and `results/summary_results.csv` (aggregated metrics).
6. Save 5 high-resolution dual-panel plots in `results/plots/`.
7. Print a formatted console summary table.

---

## Output Files

Upon completion, all outputs are saved in the `results/` directory:

```
results/
├── experiment_results.csv        # 50 rows: seed-level results for all runs
├── summary_results.csv           # 10 rows: aggregated statistics per configuration
└── plots/
    ├── baseline_convergence.png      # Dual-panel: full evolution and feasible phase (Seed 42)
    ├── selection_comparison.png      # Tournament k=2,3,4 vs Rank vs Shifted Roulette
    ├── crossover_comparison.png      # Two-point vs Uniform vs One-point crossover
    ├── mutation_comparison.png       # Mutation rate study (pm=0.01, 0.02, 0.05, 0.10)
    └── final_fitness_comparison.png  # Bar charts of feasible value and hard-penalty fitness
```

---

## Operator Handling Details

### Shifted Roulette Selection
Because the hard penalty assigns severe negative values to infeasible individuals ($\approx -10^7$ to $-10^8$), standard roulette selection cannot compute valid probabilities. We address this without altering the true objective:
$$\text{weight}_i = \text{fitness}_i - \min_{j}(\text{fitness}_j) + 1.0$$
These shifted non-negative weights are used solely to sample parents with replacement during reproduction. Each individual's stored fitness remains the exact original hard-penalty value.

### Linear Rank Selection
Individuals are sorted in ascending order of fitness ($1 = \text{worst}$, $N = \text{best}$). Selection weights are assigned directly proportional to rank ($w_i = \text{rank}_i$), establishing constant selection pressure independent of absolute fitness magnitudes.

### Real Elitism ($E=2$)
In every generation, the top $E=2$ individuals are cloned via deepcopy and preserved directly into the next generation. Offspring generation produces $N - E = 148$ individuals. This guarantees monotonic, non-decreasing best fitness across generations.
