Distributed Systems Programming Assignment 1

## Directory Structure

Assignment 1/
├── tools/
│   ├── simulate_assignment.py   # Core simulation implementation
│   └── run_experiments.py      # Experiment runner
├── VC-2210110206.cpp          # MPI Vector Clock implementation
├── SK-2210110206.cpp          # MPI SK implementation
└── inp-params.txt            # Generated during runtime

1. Install MPI:

# On Ubuntu/WSL
sudo apt-get update
sudo apt-get install mpich

## Compilation

mpic++ -std=c++17 VC-2210110206.cpp -o VC-2210110206

mpic++ -std=c++17 SK-2210110206.cpp -o SK-2210110206

## Running the Programs

python3 tools/run_experiments.py

## Input Format
The `inp-params.txt` file format:

n lambda alpha m
neighbors_of_process_1
neighbors_of_process_2
...
neighbors_of_process_n

where:
- n: number of processes
- lambda: message rate parameter
- alpha: internal event probability parameter
- m: number of messages per process
- neighbors: space-separated list of process IDs

## Output and Results

### Log Files
  - `common_log_VC.txt`: Vector Clock logs
  - `common_log_SK.txt`: SK algorithm logs

### Results Format
The experiment results show:
- Number of processes (n)
- Average entries per message for VC
- Average entries per message for SK
- Reduction percentage