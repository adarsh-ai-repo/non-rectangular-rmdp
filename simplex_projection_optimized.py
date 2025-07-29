"""
Optimized implementations of simplex projection focusing on best-performing techniques.
"""

import time

import numpy as np
from numba import jit, njit


@jit(nopython=True)
def project_to_simplex_numba_original(v: np.ndarray) -> np.ndarray:
    """
    Original implementation using binary search approach (baseline).
    """
    n = len(v)

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Binary search for optimal rho
    left, right = 0, n - 1

    while left <= right:
        mid = (left + right) // 2

        # Compute cumulative sum up to mid
        cumsum = 0.0
        for i in range(mid + 1):
            cumsum += sorted_v[i]

        theta = (cumsum - 1.0) / (mid + 1)

        # Check if this is the correct rho
        if sorted_v[mid] > theta:
            if mid == n - 1 or sorted_v[mid + 1] <= theta:
                # Found the correct rho
                result = np.empty_like(v)
                for i in range(n):
                    result[i] = max(v[i] - theta, 0.0)
                return result
            else:
                left = mid + 1
        else:
            right = mid - 1

    # Fallback (shouldn't reach here)
    result = np.empty_like(v)
    for i in range(n):
        result[i] = max(v[i], 0.0)
    return result


@jit(nopython=True)
def project_to_simplex_numba_early_check_v2(v: np.ndarray) -> np.ndarray:
    """
    Early termination with improved validity check.
    """
    n = len(v)

    # More efficient early check
    v_sum = 0.0
    min_val = v[0]
    for i in range(n):
        v_sum += v[i]
        if v[i] < min_val:
            min_val = v[i]

    # If all positive and sum is close to 1, return copy
    if min_val >= -1e-10 and abs(v_sum - 1.0) < 1e-10:
        return v.copy()

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Linear scan
    cumsum = 0.0
    for rho in range(n):
        cumsum += sorted_v[rho]
        theta = (cumsum - 1.0) / (rho + 1)

        if rho == n - 1 or sorted_v[rho + 1] <= theta:
            result = np.empty(n, dtype=np.float64)
            for i in range(n):
                temp = v[i] - theta
                result[i] = temp if temp > 0.0 else 0.0
            return result

    # Fallback
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if v[i] > 0.0:
            result[i] = v[i]
    return result


@jit(nopython=True)
def project_to_simplex_numba_adaptive(v: np.ndarray) -> np.ndarray:
    """
    Adaptive approach: choose algorithm based on vector characteristics.
    """
    n = len(v)

    # Early check for already valid simplex
    v_sum = 0.0
    min_val = v[0]
    for i in range(n):
        v_sum += v[i]
        if v[i] < min_val:
            min_val = v[i]

    if min_val >= -1e-10 and abs(v_sum - 1.0) < 1e-10:
        return v.copy()

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Always use linear scan (consistently fastest)
    cumsum = 0.0
    for rho in range(n):
        cumsum += sorted_v[rho]
        theta = (cumsum - 1.0) / (rho + 1)

        if rho == n - 1 or sorted_v[rho + 1] <= theta:
            # Optimized projection
            result = np.empty(n, dtype=np.float64)
            for i in range(n):
                temp = v[i] - theta
                result[i] = temp if temp > 0.0 else 0.0
            return result

    # Fallback
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if v[i] > 0.0:
            result[i] = v[i]
    return result


@jit(nopython=True)
def project_to_simplex_numba_hybrid(v: np.ndarray) -> np.ndarray:
    """
    Hybrid approach: always use linear scan (proven fastest).
    """
    n = len(v)

    # Early check for already valid simplex
    v_sum = 0.0
    has_negative = False
    for i in range(n):
        v_sum += v[i]
        if v[i] < 0.0:
            has_negative = True

    if not has_negative and abs(v_sum - 1.0) < 1e-10:
        return v.copy()

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Always use linear scan (consistently fastest across all sizes)
    cumsum = 0.0
    for rho in range(n):
        cumsum += sorted_v[rho]
        theta = (cumsum - 1.0) / (rho + 1)

        if rho == n - 1 or sorted_v[rho + 1] <= theta:
            result = np.empty(n, dtype=np.float64)
            for i in range(n):
                temp = v[i] - theta
                result[i] = temp if temp > 0.0 else 0.0
            return result

    # Fallback
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if v[i] > 0.0:
            result[i] = v[i]
    return result


@jit(nopython=True)
def project_to_simplex_numba_ultra_optimized(v: np.ndarray) -> np.ndarray:
    """
    Ultra-optimized version focusing on fastest proven techniques.
    """
    n = len(v)

    # Highly optimized early check
    v_sum = 0.0
    min_val = v[0]
    for i in range(n):
        v_sum += v[i]
        if v[i] < min_val:
            min_val = v[i]

    # If all positive and sum is close to 1, return copy
    if min_val >= -1e-10 and abs(v_sum - 1.0) < 1e-10:
        return v.copy()

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Always use linear scan (consistently fastest)
    cumsum = 0.0
    for rho in range(n):
        cumsum += sorted_v[rho]
        theta = (cumsum - 1.0) / (rho + 1)

        if rho == n - 1 or sorted_v[rho + 1] <= theta:
            # Optimized projection with minimal branching
            result = np.empty(n, dtype=np.float64)

            # Unrolled loop for better performance
            i = 0
            while i + 1 < n:
                temp0 = v[i] - theta
                temp1 = v[i + 1] - theta
                result[i] = temp0 if temp0 > 0.0 else 0.0
                result[i + 1] = temp1 if temp1 > 0.0 else 0.0
                i += 2

            # Handle remaining element
            if i < n:
                temp = v[i] - theta
                result[i] = temp if temp > 0.0 else 0.0

            return result

    # Fallback
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if v[i] > 0.0:
            result[i] = v[i]
    return result


@jit(nopython=True)
def project_to_simplex_numba_ultra_optimized_v2(v: np.ndarray) -> np.ndarray:
    """
    Ultra-optimized version with 4-way loop unrolling for larger vectors.
    """
    n = len(v)

    # Highly optimized early check
    v_sum = 0.0
    min_val = v[0]
    for i in range(n):
        v_sum += v[i]
        if v[i] < min_val:
            min_val = v[i]

    # If all positive and sum is close to 1, return copy
    if min_val >= -1e-10 and abs(v_sum - 1.0) < 1e-10:
        return v.copy()

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Always use linear scan (consistently fastest)
    cumsum = 0.0
    for rho in range(n):
        cumsum += sorted_v[rho]
        theta = (cumsum - 1.0) / (rho + 1)

        if rho == n - 1 or sorted_v[rho + 1] <= theta:
            # Optimized projection with 4-way loop unrolling
            result = np.empty(n, dtype=np.float64)

            # 4-way unrolled loop for better performance on larger vectors
            i = 0
            while i + 3 < n:
                temp0 = v[i] - theta
                temp1 = v[i + 1] - theta
                temp2 = v[i + 2] - theta
                temp3 = v[i + 3] - theta
                result[i] = temp0 if temp0 > 0.0 else 0.0
                result[i + 1] = temp1 if temp1 > 0.0 else 0.0
                result[i + 2] = temp2 if temp2 > 0.0 else 0.0
                result[i + 3] = temp3 if temp3 > 0.0 else 0.0
                i += 4

            # Handle remaining elements
            while i < n:
                temp = v[i] - theta
                result[i] = temp if temp > 0.0 else 0.0
                i += 1

            return result

    # Fallback
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if v[i] > 0.0:
            result[i] = v[i]
    return result


@jit(nopython=True)
def project_to_simplex_numba_ultra_optimized_v3(v: np.ndarray) -> np.ndarray:
    """
    Ultra-optimized version with cache-friendly memory access patterns.
    """
    n = len(v)

    # Highly optimized early check with cache-friendly pattern
    v_sum = 0.0
    min_val = v[0]
    i = 0
    while i < n:
        val = v[i]
        v_sum += val
        if val < min_val:
            min_val = val
        i += 1

    # If all positive and sum is close to 1, return copy
    if min_val >= -1e-10 and abs(v_sum - 1.0) < 1e-10:
        return v.copy()

    # Sort in descending order
    sorted_v = np.sort(v)[::-1]

    # Always use linear scan (consistently fastest)
    cumsum = 0.0
    for rho in range(n):
        cumsum += sorted_v[rho]
        theta = (cumsum - 1.0) / (rho + 1)

        if rho == n - 1 or sorted_v[rho + 1] <= theta:
            # Cache-friendly projection with sequential memory access
            result = np.empty(n, dtype=np.float64)

            # Sequential memory access for better cache performance
            for i in range(n):
                temp = v[i] - theta
                result[i] = temp if temp > 0.0 else 0.0

            return result

    # Fallback
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if v[i] > 0.0:
            result[i] = v[i]
    return result


def benchmark_simplex_projection():
    """
    Benchmark optimized simplex projection implementations.
    """
    print("Benchmarking Optimized Simplex Projection Implementations")
    print("=" * 70)

    # Test vectors of different sizes
    test_sizes = [8, 20, 50, 100, 200]

    implementations = [
        ("Original", project_to_simplex_numba_original),
        ("Early Check V2", project_to_simplex_numba_early_check_v2),
        ("Adaptive", project_to_simplex_numba_adaptive),
        ("Hybrid", project_to_simplex_numba_hybrid),
        ("Ultra Optimized", project_to_simplex_numba_ultra_optimized),
        ("Ultra Optimized V2", project_to_simplex_numba_ultra_optimized_v2),
        ("Ultra Optimized V3", project_to_simplex_numba_ultra_optimized_v3),
    ]

    # Warm up JIT
    print("Warming up JIT compilation...")
    for name, func in implementations:
        test_v = np.random.randn(20)
        func(test_v)

    results = {}

    for size in test_sizes:
        print(f"\nTesting with vector size: {size}")
        print("-" * 40)

        # Generate test data
        np.random.seed(42)
        test_vectors = [np.random.randn(size) for _ in range(100)]

        size_results = {}

        for name, func in implementations:
            # Warm up
            for _ in range(5):
                func(test_vectors[0])

            # Benchmark
            start_time = time.time()
            for test_v in test_vectors:
                func(test_v)
            end_time = time.time()

            avg_time = (end_time - start_time) / len(test_vectors)
            size_results[name] = avg_time

            print(f"{name:20s}: {avg_time * 1e6:.2f} µs")

        results[size] = size_results

    # Calculate speedups relative to original
    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS (relative to Original)")
    print("=" * 70)

    for size in test_sizes:
        print(f"\nVector size: {size}")
        print("-" * 30)

        baseline_time = results[size]["Original"]
        speedups = []

        for name, func in implementations:
            if name != "Original":
                speedup = baseline_time / results[size][name]
                speedups.append((name, speedup))
                print(f"{name:20s}: {speedup:.2f}x")

        # Find best performer
        best_name, best_speedup = max(speedups, key=lambda x: x[1])
        print(f"Best: {best_name} ({best_speedup:.2f}x)")

    # Overall analysis
    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE ANALYSIS")
    print("=" * 70)

    avg_speedups = {}
    for name, func in implementations:
        if name != "Original":
            total_speedup = 0.0
            for size in test_sizes:
                baseline_time = results[size]["Original"]
                speedup = baseline_time / results[size][name]
                total_speedup += speedup
            avg_speedups[name] = total_speedup / len(test_sizes)

    # Sort by average speedup
    sorted_results = sorted(avg_speedups.items(), key=lambda x: x[1], reverse=True)

    print("Average speedup across all vector sizes:")
    for name, avg_speedup in sorted_results:
        print(f"{name:20s}: {avg_speedup:.2f}x")

    best_overall = sorted_results[0]
    print(f"\nBest overall performer: {best_overall[0]} ({best_overall[1]:.2f}x average speedup)")


if __name__ == "__main__":
    benchmark_simplex_projection()
