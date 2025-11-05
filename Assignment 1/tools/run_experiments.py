#!/usr/bin/env python3
"""Compare VC and SK algorithms for different process counts."""
import subprocess
import sys
import re
from pathlib import Path

def write_inp(n, m=50):
    """Write inp-params.txt with experiment parameters."""
    # Fixed parameters for all runs
    lam = 2.0
    alpha = 0.2
    
    lines = [f"{n} {lam} {alpha} {m}\n"]
    
    # Fully connected topology - each process connects to all others
    for i in range(1, n+1):
        neighs = [str(j) for j in range(1, n+1) if j != i]
        lines.append(' '.join(neighs) + '\n')
    with open('inp-params.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines)

def run_algorithm(n, algo):
    """Run simulation for given algorithm and parse average entries."""
    write_inp(n)
    subprocess.run([sys.executable, 'tools/simulate_assignment.py', algo, 'inp-params.txt'],
                  stdout=subprocess.PIPE)
    logfile = f'common_log_{algo}.txt'
    total = count = 0
    with open(logfile, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'sent_entries=(\d+)', line)
            if m:
                total += int(m.group(1))
                count += 1
    return total/count if count else 0

print("No. of Processes (n), Standard VC, SK VC Entries (Avg of 3 runs)")
print("Running experiments (3 runs for each n)...")

# Results storage for both algorithms
results = {
    'VC': {n: [] for n in range(10, 16)},
    'SK': {n: [] for n in range(10, 16)}
}

# Run each n value 3 times for both algorithms
for run in range(3):
    print(f"\nRun {run + 1}/3:")
    for n in range(10, 16):
        print(f"  n={n}...", end='', flush=True)
        
        # Run VC
        vc_avg = run_algorithm(n, 'VC')
        results['VC'][n].append(vc_avg)
        
        # Run SK
        sk_avg = run_algorithm(n, 'SK')
        results['SK'][n].append(sk_avg)
        
        print(f" VC={vc_avg:.2f}, SK={sk_avg:.2f}")

print("\nFinal Results:")
print("n, VC (avg), SK (avg), Reduction %")

# Calculate and print averages for each n
for n in range(10, 16):
    vc_runs = results['VC'][n]
    sk_runs = results['SK'][n]
    avg_vc = sum(vc_runs) / len(vc_runs)
    avg_sk = sum(sk_runs) / len(sk_runs)
    reduction = ((avg_vc - avg_sk) / avg_vc) * 100
    print(f"{n}, {avg_vc:.2f}, {avg_sk:.2f}, {reduction:.1f}%")