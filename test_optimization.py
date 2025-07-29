import time

import numpy as np

import brute_force
import brute_force_optimized
from datamodels import PMDerivedValues, PMRandomComponents, PMUserParameters


def create_test_data(S: int = 20, A: int = 8):
    """Create test data for performance comparison."""
    params = PMUserParameters(S=S, A=A, beta=0.1, gamma=0.9, tolerance=1e-4)
    random_components = PMRandomComponents.generate(S, A, params.beta)

    # Use the proper calculate method to create derived values
    derived_values = PMDerivedValues.calculate(params, random_components)

    return params, random_components, derived_values


def test_compute_return():
    """Test compute_return function performance."""
    print("Testing compute_return function...")

    params, random_components, derived_values = create_test_data()
    P = random_components.P

    # Warm up
    for _ in range(5):
        brute_force.compute_return(P, params, random_components, derived_values)
        brute_force_optimized.compute_return(P, params, random_components, derived_values)

    # Time original function
    start_time = time.time()
    result_original = 0.0
    for _ in range(100_000):
        result_original = brute_force.compute_return(P, params, random_components, derived_values)
    original_time = time.time() - start_time

    # Time optimized function
    start_time = time.time()
    result_optimized = 0.0
    for _ in range(100_000):
        result_optimized = brute_force_optimized.compute_return(P, params, random_components, derived_values)
    optimized_time = time.time() - start_time

    print(f"Original result: {result_original:.6f}")
    print(f"Optimized result: {result_optimized:.6f}")
    print(f"Results match: {np.allclose(result_original, result_optimized)}")
    print(f"Original time: {original_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s")
    print(f"Speedup: {original_time / optimized_time:.2f}x")
    print()


def test_project_to_simplex():
    """Test project_to_simplex function performance with multiple implementations."""
    print("Testing project_to_simplex function...")

    # Test different vector sizes
    sizes = [10, 100, 1000]
    num_iterations = 1000

    for size in sizes:
        print(f"\nVector size: {size}")
        print("-" * 30)

        v = np.random.randn(size)

        # Warm up all implementations
        for _ in range(5):
            brute_force.project_to_simplex(v)
            brute_force_optimized.project_to_simplex_numba(v)

        # Time original function
        start_time = time.time()
        result_original = np.zeros_like(v)
        for _ in range(num_iterations):
            result_original = brute_force.project_to_simplex(v)
        original_time = time.time() - start_time

        # Time first numba implementation
        start_time = time.time()
        result_numba = np.zeros_like(v)
        for _ in range(num_iterations):
            result_numba = brute_force_optimized.project_to_simplex_numba(v)
        numba_time = time.time() - start_time
        # Check accuracy
        match_numba = np.allclose(result_original, result_numba, rtol=1e-10)

        print(f"Original result sum: {np.sum(result_original):.6f}")
        print(f"Numba v1 result sum: {np.sum(result_numba):.6f}")
        print(f"Numba v1 matches original: {match_numba}")
        print(f"Original time: {original_time:.4f}s")
        print(f"Numba v1 time: {numba_time:.4f}s")
        print(f"Numba v1 speedup: {original_time / numba_time:.2f}x")


def test_rpe_brute_force():
    """Test RPE_Brute_Force function performance."""
    print("Testing RPE_Brute_Force function...")

    params, random_components, derived_values = create_test_data(S=16, A=8)
    num_samples = 10_000

    # Time original function
    start_time = time.time()
    result_original = brute_force.RPE_Brute_Force(params, random_components, num_samples, derived_values)
    original_time = time.time() - start_time

    # Time optimized sequential function
    start_time = time.time()
    result_sequential = brute_force_optimized.RPE_Brute_Force_Numba(
        params, random_components, num_samples, derived_values
    )
    sequential_time = time.time() - start_time
    print("-" * 60)
    print(f"Original result: {result_original:.6f}")
    print(f"Optimized sequential result: {result_sequential:.6f}")
    print(f"Original time: {original_time:.4f}s")
    print(f"Optimized sequential time: {sequential_time:.4f}s")
    print(f"Sequential speedup: {original_time / sequential_time:.2f}x")
    print()


if __name__ == "__main__":
    print("Performance Comparison: Original vs Numba-Optimized Functions")
    print("=" * 60)

    test_compute_return()
    test_project_to_simplex()
    test_rpe_brute_force()
