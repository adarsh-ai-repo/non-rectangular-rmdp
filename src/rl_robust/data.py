from typing import List
import polars as pl
import numpy as np
from pathlib import Path


class ExperimentData:
    def __init__(self):
        self.results_df = pl.DataFrame(
            schema={
                "state_size": pl.Int64,
                "algorithm": pl.Utf8,
                "time_taken": pl.Float64,
                "penalty": pl.Float64,
            }
        )

    def add_result(
        self, state_size: int, algorithm: str, time_taken: float, penalty: float
    ) -> None:
        """Add a single result to the dataframe"""
        new_row = pl.DataFrame(
            {
                "state_size": [state_size],
                "algorithm": [algorithm],
                "time_taken": [time_taken],
                "penalty": [penalty],
            }
        )
        self.results_df = pl.concat([self.results_df, new_row])

    def save_results(self, output_dir: Path, filename: str = "results.parquet") -> None:
        """Save results to parquet file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.results_df.write_parquet(output_dir / filename)

    def load_results(self, output_dir: Path, filename: str = "results.parquet") -> None:
        """Load results from parquet file"""
        file_path = output_dir / filename
        if file_path.exists():
            self.results_df = pl.read_parquet(file_path)

    def get_algorithm_performance(self, algorithm: str) -> pl.DataFrame:
        """Get performance metrics for a specific algorithm"""
        return self.results_df.filter(pl.col("algorithm") == algorithm)

    def get_state_size_comparison(self, state_size: int) -> pl.DataFrame:
        """Get comparison of all algorithms for a specific state size"""
        return self.results_df.filter(pl.col("state_size") == state_size)
