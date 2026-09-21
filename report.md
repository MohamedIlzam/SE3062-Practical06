# SE3062 Intelligent Systems
## Practical 06 – Genetic Algorithms for 0/1 Knapsack

**Student Name:**  
**Student ID:**  

---

### 1. Problem Summary and Dataset

The 0/1 Knapsack Problem is a classic NP-hard combinatorial optimization problem. Given a collection of $n$ items, each with a profit value $v_i \in \mathbb{N}$ and a weight $w_i \in \mathbb{N}$, and a knapsack with maximum weight capacity $C$, the objective is to find a binary selection vector $\mathbf{x} = [x_1, x_2, \dots, x_n] \in \{0, 1\}^n$ that maximizes the total profit while satisfying the capacity constraint:

$$\max \sum_{i=1}^{n} v_i x_i \quad \text{subject to} \quad \sum_{i=1}^{n} w_i x_i \le C$$

In this practical, a binary chromosome representation of length $n=100$ is used, where allele $x_i = 1$ indicates that item $i$ is selected and $x_i = 0$ indicates it is excluded.

We evaluate the algorithms using an official standardized benchmark instance from David Pisinger’s benchmark suite (`smallcoeff_pisinger.tgz`, obtained from [https://hjemmesider.diku.dk/~pisinger/codes.html](https://hjemmesider.diku.dk/~pisinger/codes.html)):
- **Instance Identifier**: `knapPI_1_100_1000_1` (from file `knapPI_1_100_1000.csv`)
- **Number of items ($n$)**: 100
- **Knapsack Capacity ($C$)**: 995
- **Known Global Optimum ($z$)**: 9147 (achieved by exactly 12 items with a combined weight of $985 \le 995$)
- **Penalty Multiplier ($M$)**: $M = 10 \times \max(values) = 10 \times 997 = 9970$

The known optimal value $z = 9147$ is kept strictly as an external ground-truth reference for computing the optimality gap and is not provided to the GA during execution.

---

### 2. GA Design

The Genetic Algorithm is implemented using Python and DEAP (`deap.base`, `deap.creator`, `deap.tools`). Constraint handling uses the quadratic-order hard-penalty fitness function specified in the practical:

$$\text{Fitness}(\mathbf{x}) = \sum_{i=1}^n v_i x_i - M \cdot \max\left(0, \sum_{i=1}^n w_i x_i - C\right)$$

Because $M = 9970$, any solution exceeding the capacity $C=995$ by even one weight unit incurs a severe penalty ($\ge 9970$), driving infeasible fitness deeply negative. For feasible solutions ($\sum w_i x_i \le C$), the penalty term is zero, and the fitness equals the actual accumulated knapsack value.

The baseline parameter configuration is summarized below:

| Parameter | Specification | Value / Detail |
| :--- | :--- | :--- |
| **Encoding** | Binary List | Length $n = 100$, initialized with $U(0, 1)$ |
| **Population Size ($N$)** | Constant | 150 individuals |
| **Generations ($G$)** | Generational Loop | 200 generations |
| **Selection** | Tournament Selection | Tournament size $k = 3$ |
| **Crossover** | Two-Point Crossover | `tools.cxTwoPoint`, probability $p_c = 0.90$ |
| **Mutation** | Bit-Flip Mutation | `tools.mutFlipBit`, per-gene probability $p_m = 0.02$ |
| **Elitism ($E$)** | Real Elitism | $E = 2$ fittest individuals preserved unchanged |
| **Penalty Multiplier ($M$)**| Hard Penalty | $M = 10 \times \max(v) = 9970$ |
| **Random Seeds** | Independent Replications | `[42, 43, 44, 45, 46]` (5 runs per configuration) |

**Real Elitism Implementation**: Rather than relying solely on post-hoc tracking, real elitism is enforced within the generational loop. In every generation, deep copies of the top $E=2$ individuals from the current population are cloned and injected directly into the next generation. Offspring generation produces the remaining $N - E = 148$ individuals via parent selection, crossover, and mutation. This mathematically guarantees that the maximum fitness never regresses.

---

### 3. Baseline Results

The baseline configuration was executed across 5 independent random seeds. All 5 runs produced feasible knapsack solutions ($weight \le 995$).

| Seed | Final Best Fitness | Feasible Value | Weight | Capacity ($C$) | Optimality Gap (%) | Gen to Baseline Median |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 7896.0 | 7896 | 994 | 995 | 13.68% | >200 |
| **43** | 8628.0 | 8628 | 932 | 995 | 5.67% | >200 |
| **44** | 8842.0 | 8842 | 954 | 995 | 3.33% | 199 |
| **45** | 8817.0 | 8817 | 908 | 995 | 3.61% | 160 |
| **46** | 9147.0 | 9147 | 985 | 995 | 0.00% *(Exact Optimum)* | 186 |
| **Summary** | **8666.0 ± 468.9** | **8666.0 ± 468.9** | **954.6 ± 36.3** | **995** | **5.26%** | **Median: 8817.0** |

#### Convergence Dynamics (Seed 42 Analysis)
Referring to `results/plots/baseline_convergence.png`:
1. **Constraint Satisfaction Phase (Generations 0–35)**: Initial randomly generated individuals select approximately 50% of the items (weight $\approx 25,000$), resulting in massive overweight violations ($\approx 24,000$) and negative fitness values around $-2.4 \times 10^8$. Over the first 35 generations, tournament selection rapidly eliminates overweight individuals, driving the best fitness from $-1.5 \times 10^8$ up to positive, feasible territory.
2. **Feasible Optimization Phase (Generations 35–200)**: Once feasibility is established, the elitist GA systematically refines item combinations. For Seed 42, the best fitness climbs in discrete plateaus: 7254 (gen 59), 7562 (gen 93), 7701 (gen 138), 7820 (gen 174), and terminates at 7896 (gen 193) with weight 994 (just 1 unit below the capacity limit of 995).
3. **Seed 46 Finding Global Optimum**: Under Seed 46, the baseline configuration discovered the exact known global optimum value of **9147** at generation 186, selecting all 12 optimal items with total weight 985.

---

### 4. Operator and Parameter Comparison

Ten controlled configurations were tested (varying strictly one factor at a time relative to baseline). Each configuration was evaluated over the same 5 seeds (`[42, 43, 44, 45, 46]`). The baseline median final best fitness is **8817.00**.

| Configuration | Varied Parameter | Best Fitness (Mean ± SD) | Median Best Fitness | Mean Feasible Value | Mean Optimality Gap (%) | Gen to Baseline Median (8817.0) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | $k=3$, 2-pt, $p_m=0.02$ | 8666.0 ± 468.9 | 8817.0 | 8666.0 | 5.26% | 181.7 (3/5 reached) |
| **Tournament $k=2$** | Selection size $k=2$ | 7466.6 ± 1105.9 | 7328.0 | 7466.6 | 18.37% | None reached (>200) |
| **Tournament $k=4$** | Selection size $k=4$ | 8716.4 ± 264.0 | 8897.0 | 8716.4 | 4.71% | 83.7 (3/5 reached) |
| **Shifted Roulette** | Proportional selection | 5620.4 ± 992.4 | 5457.0 | 5620.4 | 38.55% | None reached (>200) |
| **Linear Rank** | Rank-based selection | 7418.6 ± 937.5 | 7525.0 | 7418.6 | 18.90% | None reached (>200) |
| **1-Point Crossover** | Crossover operator | 8273.2 ± 612.3 | 8333.0 | 8273.2 | 9.55% | 168.0 (1/5 reached) |
| **Uniform Crossover**| Crossover operator | 8563.6 ± 245.2 | 8575.0 | 8563.6 | 6.38% | 116.0 (1/5 reached) |
| **Mutation $p_m=0.01$**| Per-gene rate $p_m=0.01$ | **9054.0 ± 127.8** | **9147.0** | **9054.0** | **1.02%** | **69.6 (5/5 reached)** |
| **Mutation $p_m=0.05$**| Per-gene rate $p_m=0.05$ | $-11.3\text{M} \pm 2.26\text{M}$ | $-12.4\text{M}$ | 0.0 | 100.00% | None reached (>200) |
| **Mutation $p_m=0.10$**| Per-gene rate $p_m=0.10$ | $-44.4\text{M} \pm 9.82\text{M}$ | $-49.5\text{M}$ | 0.0 | 100.00% | None reached (>200) |

---

#### Detailed Empirical Analysis

##### 1. Selection Pressure (Tournament $k=2, 3, 4$, Linear Rank, Shifted Roulette)
*(Refer to `results/plots/selection_comparison.png`)*
- **Tournament Size Scaling**: Increasing tournament size from $k=2$ to $k=3$ and $k=4$ monotonically improved solution quality. Tournament $k=4$ achieved the highest mean fitness among selection variants (8716.4 ± 264.0, gap 4.71%) and reached the baseline median fastest (mean 83.7 generations). Tournament $k=2$ exerted insufficient selection pressure to overcome random genetic drift, resulting in slower convergence and a significantly lower mean fitness of 7466.6 ± 1105.9 (none of its 5 runs reached the 8817.0 threshold).
- **Linear Rank vs Tournament**: Linear rank selection produced a mean fitness of 7418.6 ± 937.5, comparable to tournament $k=2$ (7466.6), confirming that standard linear ranking imposes mild, bounded selection pressure.
- **Shifted Roulette Selection**: Shifted roulette selection yielded the lowest solution quality (5620.4 ± 992.4, gap 38.55%). Because roulette fitnesses had to be shifted by $\approx 1.7 \times 10^8$ to remain non-negative, the relative selection probabilities between above-average and below-average individuals became nearly uniform ($w_i / \sum w \approx 1/N$). This drastically diluted selection pressure, causing the algorithm to behave similarly to random walk selection.

##### 2. Crossover Operator Comparison (1-Point, 2-Point, Uniform)
*(Refer to `results/plots/crossover_comparison.png`)*
- **Two-Point Crossover (Baseline)** achieved the highest final mean fitness (8666.0 ± 468.9), striking an effective balance between building block preservation and exploratory recombination.
- **Uniform Crossover** converged rapidly during early feasible generations (reaching the 8000+ level earlier than 2-point), but plateaued slightly earlier, ending at a mean fitness of 8563.6 ± 245.2. Its high gene-mixing disruption rate ($p_{swap} = 0.5$) prevented the fine preservation of tightly linked low-weight item subsets.
- **One-Point Crossover** performed the worst among crossover types (8273.2 ± 612.3, gap 9.55%). With only a single cut point along a 100-gene chromosome, positional bias hindered effective exchange between items separated by large indices.

##### 3. Mutation Rate Impact ($p_m = 0.01, 0.02, 0.05, 0.10$)
*(Refer to `results/plots/mutation_comparison.png` and `results/plots/final_fitness_comparison.png`)*
- **The Superiority of $p_m = 0.01$**: Lowering the mutation rate to $p_m = 0.01$ produced the best overall performance in the entire study:
  - Mean best fitness: **9054.0 ± 127.8** (optimality gap only **1.02%**).
  - Median best fitness: **9147.0** (the exact global optimum).
  - **3 out of 5 runs (seeds 42, 44, 45)** discovered the exact global optimum $z = 9147$.
  - All 5 runs reached the baseline median threshold, averaging just **69.6 generations**.
  - *Explanation*: In a 100-item knapsack problem where the optimal solution contains only 12 items, an individual must maintain $\approx 88$ zeros. At $p_m = 0.01$, an average of 1 bit flip occurs per chromosome per generation, allowing the GA to fine-tune item selections without randomly introducing overweight items that trigger catastrophic penalties.
- **Catastrophic Failure at High Mutation Rates ($p_m = 0.05, 0.10$)**:
  - At $p_m = 0.05$ (5 expected flips/chromosome) and $p_m = 0.10$ (10 expected flips/chromosome), the GA **completely failed to achieve feasibility in all 5 seeds**.
  - Final best fitnesses remained trapped at **$-11,295,955.8$** and **$-44,416,331.6$**, respectively, with **0 feasible solutions found**.
  - *Explanation*: Mutation acts symmetrically on zeros and ones. In an individual with 12 ones and 88 zeros, a 5% mutation rate flips $88 \times 0.05 = 4.4$ zeros into ones every generation, adding over 2,000 weight units on average. Because selection pressure could not counteract this destructive influx, populations never escaped the hard penalty zone.

---

### 5. Conclusion

Empirical evaluation on Pisinger benchmark `knapPI_1_100_1000_1` leads to direct, data-driven conclusions:
1. **Mutation rate is the critical governing parameter** under quadratic hard penalties. A low mutation rate of $p_m = 0.01$ is optimal, achieving a 1.02% optimality gap and finding the global optimum ($z=9147$) in 60% of runs. Mutation rates $\ge 0.05$ cause complete disruption of feasibility.
2. **Higher tournament selection pressure ($k=4$) improves convergence speed and solution quality**, reducing the optimality gap from 18.37% ($k=2$) to 4.71% ($k=4$).
3. **Shifted roulette selection fails under severe penalties**, as shifting inflates background fitness and neutralizes selection differentiation.
4. **Two-point and uniform crossovers outperform one-point crossover**, with two-point crossover achieving the best balance of schema exchange and retention for 100-item binary chromosomes.
