# Non-Rectangular RMDP Experiments

This repository contains experiments for Non-Rectangular Robust Markov Decision Processes (RMDP).

## Prerequisites

This project uses `uv` for dependency management. You can install `uv` using one of the following methods:

- macOS (using Homebrew):
  ```bash
  brew install uv
  ```
- Using pip:
  ```bash
  pip install uv
  ```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/adarsh-ai-repo/non-rectangular-rmdp.git
   cd non-rectangular-rmdp
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

Run experiments using the following command:

```bash
python main.py [start] [step] [count] [beta]
```

### Required Arguments

- `start`: Initial state size (INTEGER)
- `step`: Step size between state values (INTEGER)
- `count`: Number of values to generate (INTEGER)
- `beta`: Beta value for the RMDP (FLOAT)

### Example

```bash
python main.py 10 5 4 0.05
```

### System Information For Experiments

OS: macOS Sequoia - Version 15.4.1
Chip: Apple M2
Total Number of Cores: 8 (4 performance and 4 efficiency)
Memory: 16 GB
Memory Type: LPDDR5
