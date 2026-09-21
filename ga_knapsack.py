"""
SE3062 Intelligent Systems - Practical 06: Genetic Algorithms for 0/1 Knapsack
=============================================================================
Implementation of a Genetic Algorithm using DEAP for the 0/1 Knapsack Problem.
Benchmark Dataset: David Pisinger (smallcoeff_pisinger.tgz -> knapPI_1_100_1000.csv)

Instance Used:
    Identifier: knapPI_1_100_1000_1
    Items (n): 100
    Capacity (C): 995
    Known Optimum (z): 9147

This file contains the complete end-to-end implementation:
1. Dataset extraction and loading
2. DEAP GA setup with Hard Penalty fitness
3. Selection operators (Tournament k=2,3,4, Shifted Roulette, Linear Rank)
4. Crossover operators (1-point, 2-point, Uniform)
5. Mutation operators (Bit-flip with pm=0.01, 0.02, 0.05, 0.10)
6. Real Elitism (E=2)
7. Full experimental pipeline (10 configs x 5 seeds = 50 runs)
8. Convergence metric calculation relative to baseline median
9. Visualization plotting using Matplotlib
10. CSV results export (run-level and summary tables)
"""

import os
import tarfile
import random
import copy
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deap import base, creator, tools


# =====================================================================
# PHASE 1 & 2: DATASET LOADING AND VALIDATION
# =====================================================================

ARCHIVE_PATH = "smallcoeff_pisinger.tgz"
DATA_DIR = os.path.join("data", "pisinger")
CSV_PATH = os.path.join(DATA_DIR, "knapPI_1_100_1000.csv")
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def extract_benchmark_if_needed():
    """
    Extracts knapPI_1_100_1000.csv from smallcoeff_pisinger.tgz if not already present.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        if not os.path.exists(ARCHIVE_PATH):
            raise FileNotFoundError(
                f"Archive {ARCHIVE_PATH} not found in current directory."
            )
        print(f"Extracting knapPI_1_100_1000.csv from {ARCHIVE_PATH}...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tf:
            extracted_file = tf.extractfile("knapPI_1_100_1000.csv")
            with open(CSV_PATH, "wb") as f_out:
                f_out.write(extracted_file.read())
        print(f"Extraction complete -> {CSV_PATH}")


def load_instance(csv_path: str = CSV_PATH, instance_index: int = 1) -> Dict[str, Any]:
    """
    Parses a specific problem instance from the Pisinger knapPI CSV benchmark.

    Format of Pisinger CSV:
        knapPI_1_100_1000_1    <-- Instance name
        n 100                  <-- Number of items
        c 995                  <-- Knapsack capacity
        z 9147                 <-- Known optimal value
        time 0.00              <-- Benchmark runtime
        1,94,485,0             <-- Item line: index, value (profit), weight, optimal_flag
        ... (n item lines)
        -----                  <-- Instance separator

    Returns:
        dict containing:
            'name': str
            'n': int
            'capacity': int
            'known_optimum': int
            'values': List[int]
            'weights': List[int]
            'optimal_selection': List[int]
            'M': int (Hard penalty multiplier: 10 * max(values))
    """
    extract_benchmark_if_needed()

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    target_name = f"knapPI_1_100_1000_{instance_index}"
    start_idx = None
    for idx, line in enumerate(lines):
        if line == target_name:
            start_idx = idx
            break

    if start_idx is None:
        raise ValueError(f"Instance {target_name} not found in {csv_path}")

    # Parse header
    name = lines[start_idx]
    n_line = lines[start_idx + 1]
    c_line = lines[start_idx + 2]
    z_line = lines[start_idx + 3]

    n = int(n_line.split()[1])
    capacity = int(c_line.split()[1])
    known_optimum = int(z_line.split()[1])

    # Parse items
    values: List[int] = []
    weights: List[int] = []
    optimal_selection: List[int] = []

    item_start = start_idx + 5  # line after 'time 0.00'
    for i in range(n):
        parts = lines[item_start + i].split(",")
        # Format: item_idx, profit, weight, optimal_flag
        values.append(int(parts[1]))
        weights.append(int(parts[2]))
        optimal_selection.append(int(parts[3]))

    # Validation
    assert len(values) == 100, f"Expected 100 values, got {len(values)}"
    assert len(weights) == 100, f"Expected 100 weights, got {len(weights)}"
    assert capacity > 0, f"Capacity must be > 0, got {capacity}"

    # Calculate M = 10 * max(values) as specified in practical
    M = 10 * max(values)

    return {
        "name": name,
        "n": n,
        "capacity": capacity,
        "known_optimum": known_optimum,
        "values": values,
        "weights": weights,
        "optimal_selection": optimal_selection,
        "M": M,
    }


# =====================================================================
# PHASE 3 & 4: GA DESIGN AND OPERATORS
# =====================================================================

# Initialize DEAP creators safely (guarding against duplicate class creation)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate_individual(
    individual: List[int],
    values: List[int],
    weights: List[int],
    capacity: int,
    M: int,
) -> Tuple[float]:
    """
    Computes fitness using the practical's Hard Penalty formulation:
        Fitness(x) = value(x) - M * max(0, weight(x) - C)
        where M = 10 * max(values)

    Returns:
        tuple containing (fitness,) as required by DEAP.
    """
    total_value = 0
    total_weight = 0
    for gene, v, w in zip(individual, values, weights):
        if gene:
            total_value += v
            total_weight += w

    overweight = max(0, total_weight - capacity)
    penalty = M * overweight
    fitness = total_value - penalty
    return (float(fitness),)


def sel_roulette_shifted(individuals: List[Any], k: int) -> List[Any]:
    """
    Roulette (fitness proportional) selection adapted for non-positive fitness.

    Because the hard penalty causes infeasible individuals to have negative fitness,
    standard roulette selection would fail (probabilities must be non-negative).

    Solution:
    For parent selection ONLY, we calculate positive selection weights by shifting:
        weight_i = fitness_i - min_fitness + 1.0
    The actual individual fitness attributes remain the exact original hard-penalty
    fitness values.
    """
    fits = [ind.fitness.values[0] for ind in individuals]
    min_f = min(fits)
    shift = abs(min_f) + 1.0 if min_f <= 0 else 0.0
    weights = [f + shift for f in fits]
    return random.choices(individuals, weights=weights, k=k)


def sel_rank(individuals: List[Any], k: int) -> List[Any]:
    """
    Linear Rank Selection.

    Individuals are sorted in ascending order of fitness.
    Rank 1 = worst individual, Rank N = best individual.
    Selection probability is directly proportional to rank.
    This provides uniform selection pressure regardless of fitness scaling.
    """
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness.values[0])
    n = len(sorted_inds)
    weights = list(range(1, n + 1))
    return random.choices(sorted_inds, weights=weights, k=k)


def build_toolbox(
    instance: Dict[str, Any],
    selection_type: str = "tournament",
    tournament_k: int = 3,
    crossover_type: str = "two-point",
    pm: float = 0.02,
) -> base.Toolbox:
    """
    Builds and configures the DEAP toolbox with the requested operators.
    """
    toolbox = base.Toolbox()

    # Gene and chromosome generators
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=instance["n"],
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    toolbox.register(
        "evaluate",
        evaluate_individual,
        values=instance["values"],
        weights=instance["weights"],
        capacity=instance["capacity"],
        M=instance["M"],
    )

    # Selection operator
    if selection_type == "tournament":
        toolbox.register("select", tools.selTournament, tournsize=tournament_k)
    elif selection_type == "roulette":
        toolbox.register("select", sel_roulette_shifted)
    elif selection_type == "rank":
        toolbox.register("select", sel_rank)
    else:
        raise ValueError(f"Unknown selection operator: {selection_type}")

    # Crossover operator
    if crossover_type == "one-point":
        toolbox.register("mate", tools.cxOnePoint)
    elif crossover_type == "two-point":
        toolbox.register("mate", tools.cxTwoPoint)
    elif crossover_type == "uniform":
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    else:
        raise ValueError(f"Unknown crossover operator: {crossover_type}")

    # Mutation operator (bit-flip with per-gene mutation probability pm)
    toolbox.register("mutate", tools.mutFlipBit, indpb=pm)

    return toolbox


# =====================================================================
# PHASE 3 & 4: GA RUNNER WITH REAL ELITISM
# =====================================================================

def run_ga(
    config: Dict[str, Any],
    instance: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """
    Executes a single run of the Genetic Algorithm with REAL Elitism (E=2).

    Real Elitism:
        The top E=2 individuals from the current population are cloned (deepcopied)
        and placed directly into the next generation without modification.
        The remaining (pop_size - E) individuals are produced via selection,
        crossover, and mutation.

    Returns:
        dict containing:
            'seed': int
            'final_best_fitness': float
            'final_avg_fitness': float
            'best_feasible_value': int
            'best_feasible_weight': int
            'best_feasible_solution': List[int]
            'history_best': List[float]
            'history_avg': List[float]
            'history_feasible_best': List[float]
    """
    random.seed(seed)
    np.random.seed(seed)

    pop_size = config["population"]
    ngen = config["generations"]
    pc = config["pc"]
    elitism = config["elitism"]

    toolbox = build_toolbox(
        instance=instance,
        selection_type=config["selection"],
        tournament_k=config.get("tournament_k", 3),
        crossover_type=config["crossover"],
        pm=config["pm"],
    )

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    history_best: List[float] = []
    history_avg: List[float] = []
    history_feasible_best: List[float] = []

    global_best_feasible = {
        "value": 0,
        "weight": 0,
        "fitness": float("-inf"),
        "solution": [],
    }

    values = instance["values"]
    weights = instance["weights"]
    capacity = instance["capacity"]

    def update_feasible(individuals):
        nonlocal global_best_feasible
        for ind in individuals:
            tot_w = sum(w * g for w, g in zip(weights, ind))
            if tot_w <= capacity:
                tot_v = sum(v * g for v, g in zip(values, ind))
                if tot_v > global_best_feasible["value"]:
                    global_best_feasible = {
                        "value": tot_v,
                        "weight": tot_w,
                        "fitness": ind.fitness.values[0],
                        "solution": list(ind),
                    }

    # Record Generation 0 stats
    current_fits = [ind.fitness.values[0] for ind in pop]
    history_best.append(max(current_fits))
    history_avg.append(sum(current_fits) / len(current_fits))
    update_feasible(pop)
    history_feasible_best.append(global_best_feasible["value"])

    # Generational loop
    for gen in range(1, ngen + 1):
        # 1. Real Elitism: preserve top E=2 individuals unchanged
        elites = [toolbox.clone(ind) for ind in tools.selBest(pop, elitism)]

        # 2. Select (pop_size - E) parents for offspring production
        num_offspring = pop_size - elitism
        selected = toolbox.select(pop, num_offspring)
        offspring = [toolbox.clone(ind) for ind in selected]

        # 3. Apply Crossover
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < pc:
                orig1, orig2 = list(ind1), list(ind2)
                toolbox.mate(ind1, ind2)
                if ind1 != orig1:
                    del ind1.fitness.values
                if ind2 != orig2:
                    del ind2.fitness.values

        # 4. Apply Mutation
        for mutant in offspring:
            orig = list(mutant)
            toolbox.mutate(mutant)
            if mutant != orig and hasattr(mutant, "fitness") and mutant.fitness.valid:
                del mutant.fitness.values

        # 5. Evaluate individuals with invalidated fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        new_fits = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, new_fits):
            ind.fitness.values = fit

        # 6. Replace population: elites + offspring
        pop[:] = elites + offspring

        # 7. Record metrics
        current_fits = [ind.fitness.values[0] for ind in pop]
        history_best.append(max(current_fits))
        history_avg.append(sum(current_fits) / len(current_fits))
        update_feasible(pop)
        history_feasible_best.append(global_best_feasible["value"])

    # Final best individual
    best_ind = tools.selBest(pop, 1)[0]
    final_best_fitness = best_ind.fitness.values[0]
    final_avg_fitness = history_avg[-1]

    # In case the best individual in pop is feasible, ensure global_best matches
    best_wt = sum(w * g for w, g in zip(weights, best_ind))
    best_val = sum(v * g for v, g in zip(values, best_ind))
    if best_wt <= capacity and best_val > global_best_feasible["value"]:
        global_best_feasible = {
            "value": best_val,
            "weight": best_wt,
            "fitness": final_best_fitness,
            "solution": list(best_ind),
        }

    return {
        "seed": seed,
        "final_best_fitness": final_best_fitness,
        "final_avg_fitness": final_avg_fitness,
        "best_feasible_value": global_best_feasible["value"],
        "best_feasible_weight": global_best_feasible["weight"],
        "best_feasible_solution": global_best_feasible["solution"],
        "history_best": history_best,
        "history_avg": history_avg,
        "history_feasible_best": history_feasible_best,
    }


# =====================================================================
# PHASE 5: EXPERIMENT DEFINITIONS
# =====================================================================

EXPERIMENT_CONFIGS: List[Dict[str, Any]] = [
    # 1. Baseline
    {
        "name": "baseline",
        "description": "Baseline (Tournament k=3, 2-point, pm=0.02)",
        "selection": "tournament",
        "tournament_k": 3,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 2. Tournament k=2
    {
        "name": "tournament_k2",
        "description": "Tournament k=2, 2-point, pm=0.02",
        "selection": "tournament",
        "tournament_k": 2,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 3. Tournament k=4
    {
        "name": "tournament_k4",
        "description": "Tournament k=4, 2-point, pm=0.02",
        "selection": "tournament",
        "tournament_k": 4,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 4. Roulette
    {
        "name": "roulette",
        "description": "Roulette (Shifted), 2-point, pm=0.02",
        "selection": "roulette",
        "tournament_k": None,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 5. Rank
    {
        "name": "rank",
        "description": "Linear Rank, 2-point, pm=0.02",
        "selection": "rank",
        "tournament_k": None,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 6. Crossover 1-point
    {
        "name": "crossover_1point",
        "description": "One-point Crossover, Tournament k=3, pm=0.02",
        "selection": "tournament",
        "tournament_k": 3,
        "crossover": "one-point",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 7. Crossover Uniform
    {
        "name": "crossover_uniform",
        "description": "Uniform Crossover, Tournament k=3, pm=0.02",
        "selection": "tournament",
        "tournament_k": 3,
        "crossover": "uniform",
        "pc": 0.9,
        "pm": 0.02,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 8. Mutation pm=0.01
    {
        "name": "mutation_pm0.01",
        "description": "Mutation pm=0.01, Tournament k=3, 2-point",
        "selection": "tournament",
        "tournament_k": 3,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.01,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 9. Mutation pm=0.05
    {
        "name": "mutation_pm0.05",
        "description": "Mutation pm=0.05, Tournament k=3, 2-point",
        "selection": "tournament",
        "tournament_k": 3,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.05,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
    # 10. Mutation pm=0.10
    {
        "name": "mutation_pm0.10",
        "description": "Mutation pm=0.10, Tournament k=3, 2-point",
        "selection": "tournament",
        "tournament_k": 3,
        "crossover": "two-point",
        "pc": 0.9,
        "pm": 0.10,
        "population": 150,
        "generations": 200,
        "elitism": 2,
    },
]

SEEDS = [42, 43, 44, 45, 46]


# =====================================================================
# PHASE 7, 8, 9 & 10: EXPERIMENTS, CONVERGENCE, PLOTS, AND EXPORT
# =====================================================================

def run_experiments(instance: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes all 50 experiment runs (10 configurations x 5 random seeds).
    Computes convergence metrics, summary statistics, optimality gap,
    and returns DataFrames.
    """
    all_runs_data: List[Dict[str, Any]] = []
    history_records: Dict[str, List[Dict[str, Any]]] = {}

    print("\nStarting Controlled Experiments (10 Configurations x 5 Seeds = 50 Runs)...")
    print("=" * 75)

    for idx, config in enumerate(EXPERIMENT_CONFIGS, 1):
        cfg_name = config["name"]
        history_records[cfg_name] = []

        for seed in SEEDS:
            run_res = run_ga(config, instance, seed)
            history_records[cfg_name].append(run_res)

            all_runs_data.append({
                "configuration": cfg_name,
                "description": config["description"],
                "seed": seed,
                "selection": config["selection"],
                "tournament_k": config.get("tournament_k", "") if config["selection"] == "tournament" else "",
                "crossover": config["crossover"],
                "pc": config["pc"],
                "pm": config["pm"],
                "population": config["population"],
                "generations": config["generations"],
                "elitism": config["elitism"],
                "final_best_fitness": run_res["final_best_fitness"],
                "final_avg_fitness": run_res["final_avg_fitness"],
                "best_feasible_value": run_res["best_feasible_value"],
                "best_feasible_weight": run_res["best_feasible_weight"],
                "capacity": instance["capacity"],
                "known_optimum": instance["known_optimum"],
                "optimality_gap_percent": round(
                    ((instance["known_optimum"] - run_res["best_feasible_value"]) / instance["known_optimum"]) * 100.0,
                    4,
                ),
            })

        print(f"[{idx:2d}/10] {config['description']} complete")

    df_runs = pd.DataFrame(all_runs_data)

    # Phase 7: Calculate baseline median final fitness across its 5 seeds
    baseline_final_fits = df_runs[df_runs["configuration"] == "baseline"]["final_best_fitness"].values
    baseline_median_final_fitness = float(np.median(baseline_final_fits))
    print(f"\nBaseline Median Final Best Fitness: {baseline_median_final_fitness:.2f}")

    # Determine earliest generation each run reaches/surpasses baseline median
    gen_to_median_list: List[Any] = []
    for row in all_runs_data:
        cfg = row["configuration"]
        seed = row["seed"]
        # find matching history run
        matching_run = next(r for r in history_records[cfg] if r["seed"] == seed)
        hist = matching_run["history_best"]

        # earliest generation where best fitness >= baseline_median_final_fitness
        reached_gen = None
        for g, best_f in enumerate(hist):
            if best_f >= baseline_median_final_fitness:
                reached_gen = g
                break

        row["generation_to_baseline_median"] = reached_gen if reached_gen is not None else ">200"
        gen_to_median_list.append(reached_gen)

    df_runs["generation_to_baseline_median"] = [
        r["generation_to_baseline_median"] for r in all_runs_data
    ]

    # Save detailed runs CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    detailed_csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    df_runs.to_csv(detailed_csv_path, index=False)
    print(f"Detailed run results saved -> {detailed_csv_path}")

    # Phase 8: Compute Summary Table
    summary_rows: List[Dict[str, Any]] = []
    for config in EXPERIMENT_CONFIGS:
        cfg_name = config["name"]
        subset = df_runs[df_runs["configuration"] == cfg_name]

        best_fits = subset["final_best_fitness"].values
        feas_vals = subset["best_feasible_value"].values
        gaps = subset["optimality_gap_percent"].values

        # Convergence metric calculation
        gens = [
            r["generation_to_baseline_median"]
            for r in all_runs_data
            if r["configuration"] == cfg_name
        ]
        numeric_gens = [g for g in gens if isinstance(g, (int, float))]
        if len(numeric_gens) == len(gens):
            mean_gen_str = f"{np.mean(numeric_gens):.1f}"
        elif len(numeric_gens) > 0:
            mean_gen_str = f"{np.mean(numeric_gens):.1f} ({len(numeric_gens)}/5 reached)"
        else:
            mean_gen_str = "None reached (>200)"

        summary_rows.append({
            "configuration": cfg_name,
            "description": config["description"],
            "mean_best_fitness": round(float(np.mean(best_fits)), 2),
            "sd_best_fitness": round(float(np.std(best_fits, ddof=1)), 2),
            "median_best_fitness": round(float(np.median(best_fits)), 2),
            "mean_feasible_value": round(float(np.mean(feas_vals)), 2),
            "sd_feasible_value": round(float(np.std(feas_vals, ddof=1)), 2),
            "mean_optimality_gap_pct": round(float(np.mean(gaps)), 2),
            "mean_gen_to_baseline_median": mean_gen_str,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved -> {summary_csv_path}")

    return df_runs, df_summary, history_records


def generate_plots(history_records: Dict[str, List[Dict[str, Any]]], df_summary: pd.DataFrame):
    """
    Generates high-resolution, publication-quality plots using matplotlib.
    Uses 2-panel layouts (full evolution vs zoomed feasible phase) so that
    both the penalty-escape dynamics and operator-level differences in feasible
    knapsack value are clearly visible without scale distortion.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 10.5, "figure.autolayout": True})

    # Helper function to get mean convergence curve across 5 seeds
    def get_mean_curve(cfg_name: str) -> np.ndarray:
        runs = history_records[cfg_name]
        curves = np.array([r["history_best"] for r in runs])
        return np.mean(curves, axis=0)

    gens = np.arange(201)

    # -----------------------------------------------------------------
    # 1. Baseline Convergence Plot (Seed 42)
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    base_seed42 = next(r for r in history_records["baseline"] if r["seed"] == 42)
    
    # Left: Full Range (escaping hard penalty)
    ax1.plot(gens, base_seed42["history_best"], label="Best Fitness", color="#1f77b4", linewidth=2.0)
    ax1.plot(gens, base_seed42["history_avg"], label="Average Fitness", color="#ff7f0e", linestyle="--", linewidth=1.6)
    ax1.axhline(9147, color="#2ca02c", linestyle=":", label="Optimum (z=9147)", linewidth=1.4)
    ax1.set_title("Full Evolution (Generations 0-200)\nConstraint Satisfaction Phase")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Hard-Penalty)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Right: Feasible Convergence Phase (Gen 35-200)
    zoom_gens = gens[35:]
    ax2.plot(zoom_gens, base_seed42["history_best"][35:], label="Best Fitness", color="#1f77b4", linewidth=2.2)
    ax2.axhline(9147, color="#2ca02c", linestyle=":", label="Optimum (z=9147)", linewidth=1.4)
    ax2.set_title("Feasible Optimization Phase (Generations 35-200)\nClimbing to Final Value 7896")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness (Knapsack Value)")
    ax2.set_ylim(6500, 9300)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Baseline Convergence: Tournament k=3, 2-Point CX, pm=0.02, Pop=150, E=2 (Seed 42)", fontsize=12, fontweight="bold")
    p1 = os.path.join(PLOTS_DIR, "baseline_convergence.png")
    plt.savefig(p1)
    plt.close()
    print(f"Plot saved: {p1}")

    # -----------------------------------------------------------------
    # 2. Selection Operator Comparison
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sel_configs = [
        ("tournament_k4", "Tournament k=4", "#2ca02c", "-"),
        ("baseline", "Tournament k=3 (Baseline)", "#1f77b4", "-"),
        ("tournament_k2", "Tournament k=2", "#9467bd", "-"),
        ("rank", "Linear Rank", "#ff7f0e", "-."),
        ("roulette", "Shifted Roulette", "#d62728", "--"),
    ]
    
    # Left: Full Range
    for cfg, label, col, ls in sel_configs:
        mean_curve = get_mean_curve(cfg)
        ax1.plot(gens, mean_curve, label=label, color=col, linestyle=ls, linewidth=1.8)
    ax1.axhline(9147, color="#7f7f7f", linestyle=":", label="Optimum (z=9147)")
    ax1.set_title("Full Evolution (Generations 0-200)\nMean Best Fitness Across 5 Seeds")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Best Fitness")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Right: Feasible Convergence Phase (Gen 40-200)
    for cfg, label, col, ls in sel_configs:
        mean_curve = get_mean_curve(cfg)
        ax2.plot(gens[40:], mean_curve[40:], label=label, color=col, linestyle=ls, linewidth=2.0)
    ax2.axhline(9147, color="#7f7f7f", linestyle=":", label="Optimum (z=9147)")
    ax2.set_title("Feasible Optimization Phase (Generations 40-200)\nDirect Operator Comparison")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Best Fitness (Feasible Value)")
    ax2.set_ylim(4500, 9300)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Selection Operator Comparison: Tournament (k=2,3,4) vs Rank vs Shifted Roulette", fontsize=12, fontweight="bold")
    p2 = os.path.join(PLOTS_DIR, "selection_comparison.png")
    plt.savefig(p2)
    plt.close()
    print(f"Plot saved: {p2}")

    # -----------------------------------------------------------------
    # 3. Crossover Operator Comparison
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    cx_configs = [
        ("baseline", "Two-Point Crossover (Baseline)", "#1f77b4", "-"),
        ("crossover_uniform", "Uniform Crossover (p=0.5)", "#17becf", "-."),
        ("crossover_1point", "One-Point Crossover", "#e377c2", "--"),
    ]

    for cfg, label, col, ls in cx_configs:
        mean_curve = get_mean_curve(cfg)
        ax1.plot(gens, mean_curve, label=label, color=col, linestyle=ls, linewidth=1.8)
    ax1.axhline(9147, color="#7f7f7f", linestyle=":", label="Optimum (z=9147)")
    ax1.set_title("Full Evolution (Generations 0-200)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Best Fitness")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    for cfg, label, col, ls in cx_configs:
        mean_curve = get_mean_curve(cfg)
        ax2.plot(gens[35:], mean_curve[35:], label=label, color=col, linestyle=ls, linewidth=2.0)
    ax2.axhline(9147, color="#7f7f7f", linestyle=":", label="Optimum (z=9147)")
    ax2.set_title("Feasible Optimization Phase (Generations 35-200)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Best Fitness (Feasible Value)")
    ax2.set_ylim(7000, 9300)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Crossover Operator Comparison: Two-Point vs Uniform vs One-Point", fontsize=12, fontweight="bold")
    p3 = os.path.join(PLOTS_DIR, "crossover_comparison.png")
    plt.savefig(p3)
    plt.close()
    print(f"Plot saved: {p3}")

    # -----------------------------------------------------------------
    # 4. Mutation Rate Comparison
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    mut_configs = [
        ("mutation_pm0.01", "Mutation pm = 0.01", "#8c564b", "-"),
        ("baseline", "Mutation pm = 0.02 (Baseline)", "#1f77b4", "-"),
        ("mutation_pm0.05", "Mutation pm = 0.05", "#ff7f0e", "--"),
        ("mutation_pm0.10", "Mutation pm = 0.10", "#d62728", "-."),
    ]

    for cfg, label, col, ls in mut_configs:
        mean_curve = get_mean_curve(cfg)
        ax1.plot(gens, mean_curve, label=label, color=col, linestyle=ls, linewidth=1.8)
    ax1.axhline(9147, color="#7f7f7f", linestyle=":", label="Optimum (z=9147)")
    ax1.set_title("Full Evolution (All Rates: pm=0.01, 0.02, 0.05, 0.10)\nHigh pm Fails Constraint Satisfaction")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Best Fitness")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Zoom for feasible mutation rates (pm=0.01 and pm=0.02)
    for cfg, label, col, ls in [mut_configs[0], mut_configs[1]]:
        mean_curve = get_mean_curve(cfg)
        ax2.plot(gens[30:], mean_curve[30:], label=label, color=col, linestyle=ls, linewidth=2.2)
    ax2.axhline(9147, color="#2ca02c", linestyle=":", label="Optimum (z=9147)", linewidth=1.5)
    ax2.set_title("Feasible Optimization Phase (pm=0.01 vs pm=0.02)\npm=0.01 Reaches Near-Optimum (9054 vs 8666)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Best Fitness (Feasible Value)")
    ax2.set_ylim(7000, 9300)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Mutation Rate Comparison: Impact on Penalty Avoidance and Convergence", fontsize=12, fontweight="bold")
    p4 = os.path.join(PLOTS_DIR, "mutation_comparison.png")
    plt.savefig(p4)
    plt.close()
    print(f"Plot saved: {p4}")

    # -----------------------------------------------------------------
    # 5. Final Comparison Bar Chart (Feasible Values & Full Fitness)
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    labels = [
        "Baseline\n(k=3, 2pt, .02)",
        "Tourn\nk=2",
        "Tourn\nk=4",
        "Shifted\nRoulette",
        "Linear\nRank",
        "1-Point\nCX",
        "Uniform\nCX",
        "pm=\n0.01",
        "pm=\n0.05",
        "pm=\n0.10",
    ]
    feas_means = df_summary["mean_feasible_value"].values
    feas_sds = df_summary["sd_feasible_value"].values
    fit_means = df_summary["mean_best_fitness"].values
    fit_sds = df_summary["sd_best_fitness"].values

    x_pos = np.arange(len(labels))
    colors = ["#1f77b4"] + ["#aec7e8"] * 4 + ["#98df8a"] * 2 + ["#2ca02c", "#ff9896", "#d62728"]

    # Left: Feasible Knapsack Value (Mean +- SD)
    bars1 = ax1.bar(x_pos, feas_means, yerr=feas_sds, capsize=4, color=colors, edgecolor="#333333", alpha=0.9)
    ax1.axhline(9147, color="#2ca02c", linestyle="--", linewidth=1.5, label="Optimum (z=9147)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Best Feasible Value (Mean ± SD)")
    ax1.set_title("Best Feasible Knapsack Value Achieved\n(pm=0.05 & 0.10 yielded 0 feasible solutions)")
    ax1.set_ylim(0, 10000)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    for bar, m in zip(bars1, feas_means):
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 150, f"{m:.0f}", ha="center", va="bottom", fontsize=8)
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, 200, "0\n(Infeas)", ha="center", va="bottom", fontsize=7.5, color="#d62728")

    # Right: Hard Penalty Fitness (Log-scaled or Full scale)
    bars2 = ax2.bar(x_pos, fit_means, yerr=fit_sds, capsize=4, color=colors, edgecolor="#333333", alpha=0.9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("Final Best Fitness (Hard-Penalty)")
    ax2.set_title("Hard-Penalty Fitness Across All Configurations\n(Demonstrating Extreme Penalization at High pm)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, m in zip(bars2, fit_means):
        if m < 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, m - 3e6, f"{m/1e6:.1f}M", ha="center", va="top", fontsize=8)
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, 1e6, f"{m:.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Final Solution Quality Comparison Across All 10 Experimental Configurations", fontsize=12, fontweight="bold")
    p5 = os.path.join(PLOTS_DIR, "final_fitness_comparison.png")
    plt.savefig(p5)
    plt.close()
    print(f"Plot saved: {p5}")


# =====================================================================
# PHASE 11: MAIN ENTRY POINT
# =====================================================================

def main():
    print("=" * 75)
    print("SE3062 Intelligent Systems - Practical 06")
    print("Genetic Algorithms for 0/1 Knapsack Problem")
    print("=" * 75)

    # 1. Load and validate Pisinger instance
    instance = load_instance(CSV_PATH, instance_index=1)
    print("\nDataset Summary:")
    print(f"  Benchmark File      : {CSV_PATH}")
    print(f"  Instance Name       : {instance['name']}")
    print(f"  Number of Items (n) : {instance['n']}")
    print(f"  Knapsack Capacity(C): {instance['capacity']}")
    print(f"  Known Optimum (z)   : {instance['known_optimum']}")
    print(f"  Min/Max Item Value  : {min(instance['values'])} / {max(instance['values'])}")
    print(f"  Min/Max Item Weight : {min(instance['weights'])} / {max(instance['weights'])}")
    print(f"  Hard Penalty M      : {instance['M']} (10 * max_val)")
    print(f"  Validation Status   : PASSED (len(values)==100, len(weights)==100, C>0)\n")

    # 2. Run all experiments
    df_runs, df_summary, history_records = run_experiments(instance)

    # 3. Generate visualization plots
    print("\nGenerating Visualization Plots...")
    generate_plots(history_records, df_summary)

    # 4. Print Summary Table
    print("\n" + "=" * 100)
    print("SUMMARY RESULTS TABLE (Aggregated Across 5 Seeds: 42, 43, 44, 45, 46)")
    print("=" * 100)
    print(
        f"{'Configuration':<18} | {'Best Fitness (Mean±SD)':<24} | {'Median Fit':<10} | "
        f"{'Feasible Value':<14} | {'Gap (%)':<8} | {'Gen to Base Median':<18}"
    )
    print("-" * 100)

    for _, row in df_summary.iterrows():
        fit_str = f"{row['mean_best_fitness']:.1f} ± {row['sd_best_fitness']:.1f}"
        val_str = f"{row['mean_feasible_value']:.1f}"
        gap_str = f"{row['mean_optimality_gap_pct']:.2f}%"
        print(
            f"{row['configuration']:<18} | {fit_str:<24} | {row['median_best_fitness']:<10.1f} | "
            f"{val_str:<14} | {gap_str:<8} | {row['mean_gen_to_baseline_median']:<18}"
        )

    print("=" * 100)
    print("\nExecution completed successfully.")
    print(f"Detailed run results : {os.path.join(RESULTS_DIR, 'experiment_results.csv')}")
    print(f"Summary table        : {os.path.join(RESULTS_DIR, 'summary_results.csv')}")
    print(f"Plots directory      : {PLOTS_DIR}")


if __name__ == "__main__":
    main()