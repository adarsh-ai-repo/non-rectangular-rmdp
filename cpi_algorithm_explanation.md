# CPI Algorithm 3.2 for Robust Policy Evaluation

## Simple Explanation

The Conservative Policy Iteration (CPI) Algorithm 3.2 aims to find the worst-case performance of a given policy $\pi$ in a reinforcement learning setting when the transition dynamics (how states change) are uncertain.

Instead of assuming we know the exact transition probabilities, we acknowledge there's uncertainty in our model. The algorithm iteratively refines an estimate of the "worst-case" transition kernel $P$ within an uncertainty set $\mathcal{U}_c$. It does this by repeatedly finding the kernel $P^*$ that minimizes a specific function $f(P)$ related to the policy's performance, and then gradually updating our current estimate toward this worst-case kernel.

## Mathematical Formulation

### Key Components

1. **Value Function** $v^\pi_P(s)$: The expected discounted sum of rewards starting from state $s$ and following policy $\pi$ under transition kernel $P$. It satisfies the Bellman equation:

   $$v^\pi_P(s) = \sum_{a} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) v^\pi_P(s') \right]$$

2. **State Distribution** $d^\pi_{P_{\text{hat}}}(s|s_0)$: The discounted probability of being in state $s$ at some future time step, given starting state $s_0$ and following policy $\pi$ under the _nominal_ kernel $P_{\text{hat}}$. The overall state distribution $d^\pi_{P_{\text{hat}}}(s)$ is averaged over initial states.

3. **Advantage Function** $A^\pi_P(s,a,s')$: Measures the advantage of transitioning to state $s'$ from state-action pair $(s,a)$ compared to the average next state value:

   $$A^\pi_P(s,a,s') = \gamma\left[ P(s'|s,a)v^\pi_P(s') - \sum_{s''} P(s''|s,a)v^\pi_P(s'') \right]$$

### Algorithm Steps

1. **Define Objective Function** $f(P)$:

   $$f(P) = \frac{1}{1-\gamma}\sum_{s,a,s'} d^\pi_{P_{\text{hat}}}(s) \pi(a|s) A^\pi_{P_{\text{hat}}}(s,a,s') P(s'|s,a)$$

   This function measures the expected "disadvantage" introduced by using kernel $P$ instead of the nominal kernel $P_{\text{hat}}$, weighted by state visitation probabilities.

2. **Find Worst Kernel** $P^*$:

   $$P^* = \arg\min_{P \in \mathcal{U}_c} f(P)$$

   This is an optimization problem: find the kernel $P$ within the allowed uncertainty set $\mathcal{U}_c$ that minimizes $f(P)$.

3. **Update Rule**:

   $$P_{n+1} = (1-\alpha_n)P_n + \alpha_n P^*$$

   Where the step size is:

   $$\alpha_n = -\frac{(1-\gamma)^3}{4\gamma^2}f(P^*)$$

   This is a convex combination update, moving $P_n$ towards $P^*$. The step size $\alpha_n$ is proportional to the negative of $f(P^*)$, meaning larger negative values (worse performance under $P^*$) lead to larger steps.

4. **Convergence**: The process continues until $P_n$ converges to $P_\infty$, which represents the worst-case transition kernel within the uncertainty set.

5. **Return**: The algorithm returns the robust return $J^\pi_{P_\infty}$, which is the expected return of policy $\pi$ under the worst-case kernel $P_\infty$.

## Relation to Codebase

While the exact CPI Algorithm 3.2 isn't implemented in the codebase, several components and concepts are present:

### Data Models (`datamodels.py`)

- `PMUserParameters`: Defines $S$ (states), $A$ (actions), $\gamma$ (discount factor), $\beta$ (uncertainty radius, likely defining $\mathcal{U}_c$).
- `PMRandomComponents`: Holds the nominal kernel $P$ (equivalent to $P_{\text{hat}}$), policy $\pi$, rewards $R$.
- `PMDerivedValues`: Calculates essential quantities like:
  - `P_pi`: Policy-averaged kernel
  - `v_pi`: Value function under $P_{\text{hat}}$
  - `d_pi`: State distribution under $P_{\text{hat}}$
  - `D_pi`: Occupation matrix

The `compute_value_function` and `compute_occupation_measures` methods directly compute parts needed for the algorithm.

### Optimization Methods

The core of the CPI algorithm involves the minimization step $\arg\min_{P \in \mathcal{U}_c} f(P)$. The codebase explores different optimization techniques for related problems:

- `optimize_using_slsqp_method` (`optimize_using_scipy.py`): Uses a general-purpose constrained optimization (SLSQP).
- `optimize_using_eigen_value_and_bisection` (`algorithm_1.py`): Solves a related optimization problem using eigenvalue analysis and bisection search.
- `optimize_using_random_rank_1_kernel` (`rank_1_random_matrix.py`): Uses random sampling for a specific type of kernel perturbation (rank-1).
- `RPE_Brute_Force` (`brute_force.py`): Samples random kernels from the uncertainty set and computes the return for each, finding the minimum return (maximum penalty).

### Missing Pieces

- The explicit calculation of the advantage function $A^\pi_P$ as defined in the algorithm.
- The specific objective function $f(P)$.
- The iterative update rule for $P_n$ using $\alpha_n$.

## Summary

The CPI Algorithm 3.2 provides a principled approach to robust policy evaluation by iteratively refining an estimate of the worst-case transition kernel. While the codebase contains many necessary ingredients, it does not implement the specific iterative structure of CPI Algorithm 3.2. Instead, it tackles related problems (finding the worst-case value directly) using different strategies (SLSQP, eigenvalue methods, random sampling) rather than the iterative kernel refinement approach of CPI.
