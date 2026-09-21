# SE3062 Intelligent Systems
## Practical 06 – Genetic Algorithms for 0/1 Knapsack

**Student Name:**  
**Student ID:**  
**Date:** September 2026  

---

### 1. Problem & Dataset

The 0/1 Knapsack Problem is an NP-hard combinatorial optimization task where $n$ items, each with profit $v_i$ and weight $w_i$, must be selected via binary decisions $x_i \in \{0, 1\}$ to maximize total profit $\sum v_i x_i$ subject to capacity constraint $\sum w_i x_i \le C$. We evaluate using the official David Pisinger benchmark instance `knapPI_1_100_1000_1` ($n=100$, knapsack capacity $C=995$, and known global optimum $z=9147$ with optimal weight $W=985 \le 995$, consisting of 12 selected items). The known optimum is reserved strictly for post-run optimality gap verification and is not exposed to the genetic algorithm.

---

### 2. GA Design

The algorithm is implemented in Python with DEAP using a 100-bit chromosome representation ($1 = \text{selected}$, $0 = \text{excluded}$). Infeasible solutions exceeding capacity $C$ are penalized using the hard-penalty fitness function (a linear hard penalty with respect to capacity overflow): $\text{Fitness}(\mathbf{x}) = \sum v_i x_i - M \cdot \max(0, \sum w_i x_i - C)$, where $M = 10 \times \max(v) = 9970$. Real elitism ($E=2$) is strictly enforced each generation by cloning the two fittest individuals directly into the next population, with the remaining $N-E=148$ individuals produced via parent selection, crossover, and mutation.

| Parameter | Specification / Value | Description |
| :--- | :--- | :--- |
| **Representation** | Binary List ($n=100$) | 100 alleles initialized with $U(0, 1)$ |
| **Population Size ($N$)** | 150 | Number of individuals per generation |
| **Generations ($G$)** | 200 | Total generational cycles per run |
| **Selection** | Tournament ($k=3$) | Parent selection operator (k=2, 4, Rank, Shifted Roulette in experiments) |
| **Crossover** | Two-Point Crossover | Two cut points swapped; probability $p_c = 0.90$ |
| **Mutation** | Bit-Flip Mutation | Independent bit-flip per gene; per-gene rate $p_m = 0.02$ |
| **Elitism ($E$)** | Real Elitism ($E=2$) | Top 2 chromosomes preserved unchanged without mutation |
| **Penalty Multiplier ($M$)**| Hard Penalty ($M=9970$) | $M = 10 \times \max(values)$; heavily penalizes overweight solutions |
| **Evaluation Seeds** | 5 Independent Replications | Random seeds: `42, 43, 44, 45, 46` |

<div style="page-break-after: always;"></div>

---

### 3. Results

All 50 experimental runs (10 configurations $\times$ 5 seeds) were executed deterministically. Each run explicitly tracked whether any feasible individual was encountered via a boolean `feasible_found` flag. The baseline median final best fitness across its 5 runs is **8817.00**.

#### Baseline Convergence (Seed 42)
![Baseline Convergence Curve](results/plots/baseline_convergence.png)
*Figure 1: Baseline convergence under Seed 42. Left: Full evolution over generations 0–200 demonstrating escape from initial severe hard-penalty violations ($-2.4 \times 10^8$). Right: Zoomed view of feasible optimization (generations 35–200) climbing to a feasible value of 7896 (weight 994).*

#### Experimental Results Summary Table

| Configuration | Varied Factor | Feasible Runs | Best Fitness (Mean ± SD) | Median Fit | Feasible Value | Gap (%) | Gen to Base Median (8817.0) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | $k=3$, 2-pt, $p_m=0.02$ | 5/5 | 8666.0 ± 468.9 | 8817.0 | 8666.0 | 5.26% | 181.7 (3/5 reached) |
| **Tournament $k=2$** | Selection size $k=2$ | 5/5 | 7466.6 ± 1105.9 | 7328.0 | 7466.6 | 18.37% | None reached (>200) |
| **Tournament $k=4$** | Selection size $k=4$ | 5/5 | 8716.4 ± 264.0 | 8897.0 | 8716.4 | 4.71% | 83.7 (3/5 reached) |
| **Shifted Roulette** | Proportional selection | 5/5 | 5620.4 ± 992.4 | 5457.0 | 5620.4 | 38.55% | None reached (>200) |
| **Linear Rank** | Rank-based selection | 5/5 | 7418.6 ± 937.5 | 7525.0 | 7418.6 | 18.90% | None reached (>200) |
| **1-Point Crossover** | Single cut point | 5/5 | 8273.2 ± 612.3 | 8333.0 | 8273.2 | 9.55% | 168.0 (1/5 reached) |
| **Uniform Crossover**| Uniform swap ($p=0.5$) | 5/5 | 8563.6 ± 245.2 | 8575.0 | 8563.6 | 6.38% | 116.0 (1/5 reached) |
| **Mutation $p_m=0.01$**| Per-gene rate $p_m=0.01$ | 5/5 | **9054.0 ± 127.8** | **9147.0** | **9054.0** | **1.02%** | **69.6 (5/5 reached)** |
| **Mutation $p_m=0.05$**| Per-gene rate $p_m=0.05$ | 0/5 | $-11.3\text{M} \pm 2.26\text{M}$ | $-12.4\text{M}$ | 0.0 | 100.00% | None reached (>200) |
| **Mutation $p_m=0.10$**| Per-gene rate $p_m=0.10$ | 0/5 | $-44.4\text{M} \pm 9.82\text{M}$ | $-49.5\text{M}$ | 0.0 | 100.00% | None reached (>200) |

*Note: In the baseline configuration, Seed 46 discovered the exact global optimum (9147, weight 985). In configuration $p_m=0.01$, 3 out of 5 seeds (42, 44, 45) discovered the exact global optimum (9147).*

<div style="page-break-after: always;"></div>

---

### 4. Discussion

**Selection Pressure**: Increasing tournament size from $k=2$ to $k=4$ systematically improved final solution quality from 7466.6 ± 1105.9 ($k=2$) to 8666.0 ± 468.9 ($k=3$) and 8716.4 ± 264.0 ($k=4$). Tournament $k=4$ also converged fastest, reaching the baseline median in an average of 83.7 generations. Linear Rank selection yielded a mean fitness of 7418.6 ± 937.5, comparable to tournament $k=2$. Shifted roulette produced the lowest mean solution quality among the tested selection methods (5620.4 ± 992.4, 38.55% optimality gap). The shift was required because the hard-penalty objective can produce negative fitness values. On this benchmark and under these settings, shifted roulette was less effective than tournament selection; the experiment does not isolate a single causal mechanism for this difference.

**Crossover Operators**: Crossover operator choice affected performance on this benchmark. Two-point crossover achieved a mean best fitness of 8666.0 ± 468.9, uniform crossover achieved 8563.6 ± 245.2, and one-point crossover achieved 8273.2 ± 612.3. The empirical results show that two-point crossover produced the highest mean solution quality among the three tested operators on this instance, while the underlying schema-level mechanisms (such as positional bias or building-block disruption) were not separately measured.

**Mutation Rates**: Among the tested mutation rates, $p_m=0.01$ was the best-performing mutation rate among the tested values, achieving a mean best fitness of 9054.0 ± 127.8 (mean gap 1.02%), reaching the baseline median in an average of 69.6 generations across all five seeds, and finding the exact global optimum ($z=9147$) in 3 out of 5 runs (seeds 42, 44, 45). For $p_m = 0.05$ and $p_m = 0.10$, the rerun explicitly verified using the `feasible_found` flag that 0 out of 5 runs encountered any feasible individual throughout 200 generations (`feasible_found = False` across all runs). Consequently, both high-mutation configurations yielded zero feasible solutions and terminated with large negative penalized fitnesses ($-11.3\text{M}$ and $-44.4\text{M}$, respectively).

---

### 5. Conclusion

On this Pisinger instance (`knapPI_1_100_1000_1`), under the tested GA configuration across the five evaluation seeds, mutation rate had the most pronounced effect on outcome: $p_m = 0.01$ was the best-performing rate among tested values (1.02% mean gap, reaching the global optimum $z=9147$ in 3 of 5 runs), whereas $p_m \ge 0.05$ failed to find any feasible individual in all five runs. Among selection methods, tournament selection ($k=3, 4$) achieved higher final fitness than linear rank and shifted roulette. Among crossover operators, 2-point crossover yielded the highest mean performance (8666.0), followed by uniform (8563.6) and 1-point (8273.2). These findings are specific to this problem instance and parameter set.
