import os
import sqlite3
from typing import Any, Dict

from datamodels import AlgorithmPerformanceData, initialize_empty_performance_data


def initialize_database(db_path: str) -> None:
    """
    Initialize the SQLite database with the required schema.

    Args:
        db_path: Path to the SQLite database file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with hash, algorithm_name, and iteration_count as a composite primary key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS performance_data (
        hash TEXT NOT NULL,
        algorithm_name TEXT NOT NULL,
        iteration_count INTEGER NOT NULL,
        time_taken REAL,
        j_pi REAL,
        nominal_return REAL,
        S INTEGER,
        A INTEGER,
        beta REAL,
        start_time TEXT,
        PRIMARY KEY (hash, algorithm_name, iteration_count)
    )
    """)

    conn.commit()
    conn.close()


def save_performance_data(db_path: str, data: AlgorithmPerformanceData) -> None:
    """
    Save performance data to SQLite database.

    Args:
        db_path: Path to the SQLite database file
        data: Performance data to save
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert data
    for i in range(len(data)):
        # Convert datetime to ISO format string for SQLite storage

        try:
            cursor.execute(
                """
            INSERT OR REPLACE INTO performance_data 
            (hash, algorithm_name, iteration_count, time_taken, j_pi, nominal_return, S, A, beta, start_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["hash"][i],
                    data["algorithm_name"][i],
                    data["iteration_count"][i],
                    data["time_taken"][i],
                    data["j_pi"][i],
                    data["nominal_return"][i],
                    data["S"][i],
                    data["A"][i],
                    data["beta"][i],
                    data["start_time"][i],
                ),
            )
        except sqlite3.IntegrityError:
            # If there's a primary key conflict, update the existing record
            cursor.execute(
                """
            UPDATE performance_data 
            SET time_taken = ?, j_pi = ?, S = ?, A = ?, beta = ?, start_time = ?, nominal_return = ?
            WHERE hash = ? AND algorithm_name = ? AND iteration_count = ?
            """,
                (
                    data["time_taken"][i],
                    data["j_pi"][i],
                    data["S"][i],
                    data["A"][i],
                    data["beta"][i],
                    data["start_time"][i],
                    data["nominal_return"][i],
                    data["hash"][i],
                    data["algorithm_name"][i],
                    data["iteration_count"][i],
                ),
            )

    conn.commit()
    conn.close()


def load_performance_data(db_path: str, conditions: Dict[str, Any] = None) -> AlgorithmPerformanceData:
    """
    Load performance data from SQLite database with optional filtering.

    Args:
        db_path: Path to the SQLite database file
        conditions: Optional dictionary of column-value pairs to filter results

    Returns:
        AlgorithmPerformanceData: The loaded performance data
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM performance_data"
    params = []

    if conditions:
        where_clauses = []
        for key, value in conditions.items():
            where_clauses.append(f"{key} = ?")
            params.append(value)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

    cursor.execute(query, params)
    rows = cursor.fetchall()

    result: AlgorithmPerformanceData = initialize_empty_performance_data()
    for row in rows:
        result["algorithm_name"].append(row["algorithm_name"])
        result["iteration_count"].append(row["iteration_count"])
        result["time_taken"].append(row["time_taken"])
        result["j_pi"].append(row["j_pi"])
        result["S"].append(row["S"])
        result["A"].append(row["A"])
        result["beta"].append(row["beta"])
        result["hash"].append(row["hash"])
        result["start_time"].append(row["start_time"])
        result["nominal_return"].append(row["nominal_return"])

    conn.close()
    return result
