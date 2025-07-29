"""
Test to verify that the optimized simplex projection functions produce the same results as the original.
"""

import numpy as np

from brute_force_optimized import project_to_simplex_numba
from simplex_projection_optimized import project_to_simplex_numba_ultra_optimized_v2


def test_simplex_projection_correctness():
    """Test that optimized functions produce same results as original."""
    print("Testing Simplex Projection Correctness")
    print("=" * 50)

    # Test cases
    test_cases = [
        # Simple cases
        np.array([1.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.33, 0.33, 0.34]),
        # Cases needing projection
        np.array([2.0, 1.0, 0.5]),
        np.array([-1.0, 2.0, 0.5]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        # Random cases
        np.random.randn(8),
        np.random.randn(20),
        np.random.randn(50),
        np.random.randn(100),
        np.random.randn(200),
        # Edge cases
        np.array([1.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([-1.0, -1.0, -1.0, 4.0]),
        np.ones(10) * 0.1,
        np.ones(50) * 0.02,
    ]

    tolerance = 1e-10
    all_passed = True

    for i, test_case in enumerate(test_cases):
        print(f"Test case {i + 1}: size={len(test_case)}")

        # Get results from both functions
        result_original = project_to_simplex_numba(test_case.copy())
        result_optimized = project_to_simplex_numba_ultra_optimized_v2(test_case.copy())

        # Check if results are close
        if not np.allclose(result_original, result_optimized, atol=tolerance):
            print(f"  ❌ FAILED: Results differ")
            print(f"    Original:  {result_original}")
            print(f"    Optimized: {result_optimized}")
            print(f"    Max diff:  {np.max(np.abs(result_original - result_optimized))}")
            all_passed = False
        else:
            print(f"  ✅ PASSED: Results match within tolerance")

        # Verify both results are valid simplexes
        def is_valid_simplex(x):
            return np.all(x >= -tolerance) and abs(np.sum(x) - 1.0) < tolerance

        if not is_valid_simplex(result_original):
            print(f"  ❌ FAILED: Original result not a valid simplex")
            print(f"    Sum: {np.sum(result_original)}, Min: {np.min(result_original)}")
            all_passed = False

        if not is_valid_simplex(result_optimized):
            print(f"  ❌ FAILED: Optimized result not a valid simplex")
            print(f"    Sum: {np.sum(result_optimized)}, Min: {np.min(result_optimized)}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL TESTS PASSED: Optimized function produces identical results!")
        return True
    else:
        print("❌ SOME TESTS FAILED: Optimized function produces different results!")
        return False


if __name__ == "__main__":
    test_simplex_projection_correctness()
