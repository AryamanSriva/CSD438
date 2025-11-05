#!/usr/bin/env python3
"""
Simulate Vector Clock algorithms locally.
"""
import sys
import threading
import time
import random
from queue import Queue, Empty
from pathlib import Path


class Process(threading.Thread):
    def __init__(self, pid, n, adj, lam, alpha, m, algorithm, logs, stop_event):
        super().__init__()
        self.pid = pid
        self.n = n
        self.adj = adj
        self.lam = lam
        self.alpha = alpha
        self.m = m
        self.algo = algorithm
        self.inbox = Queue()
        self.vc = [0]*n
        self.last_sent = {nei: [-1]*n for nei in range(n)}
        self.sent_count = 0
        self.internal_count = 0
        self.send_count = 0
        self.logs = logs
        self.stop_event = stop_event
        self.rng = random.Random()
        self.last_update_time = [0] * n
        self.current_time = 0

    def log(self, s):
        # timestamp in milliseconds since epoch for sorting
        ms = int(time.time() * 1000)
        t = time.strftime('%H:%M:%S', time.localtime())
        # print vector with trailing space inside bracket to match assignment format
        vc_str = ' '.join(str(x) for x in self.vc) + ' '
        # prefix with ms so we can sort across processes later
        self.logs.append(f"{ms} Process{self.pid+1} {s} at {t}, vc: [{vc_str}]")

    def run(self):
        p_internal = self.alpha / (self.alpha + 1.0)
        while not self.stop_event.is_set():
            # Process all incoming messages first
            while True:
                try:
                    msg = self.inbox.get_nowait()
                    if self.algo == 'VC':
                        sender, msgid, vec = msg
                        for i in range(self.n):
                            self.vc[i] = max(self.vc[i], vec[i])
                        self.log(f"receives {msgid} from Process{sender+1}")
                    else:  # SK
                        sender, msgid, idx_vals = msg
                        for idx, val in idx_vals.items():
                            if val > self.vc[idx]:
                                self.vc[idx] = val
                        self.log(f"receives {msgid} from Process{sender+1}")
                except Empty:
                    break

            # check termination condition
            if self.sent_count >= self.m:
                # done sending; wait a bit to drain inbox
                time.sleep(0.05)
                # if inbox empty, set stop
                if self.inbox.empty():
                    break

            # sleep for inter-event time (exponential)
            delay = self.rng.expovariate(1.0/self.lam) if self.lam>0 else 0.0
            time.sleep(delay/1000.0)  # lam is in ms in original; keep small

            # choose internal or send
            if self.rng.random() < p_internal:
                # internal
                self.vc[self.pid] += 1
                self.internal_count += 1
                eid = f"e{self.pid+1}{self.internal_count}"
                self.log(f"executes internal event {eid}")
            else:
                # send
                if not self.adj[self.pid]:
                    self.vc[self.pid] += 1
                    self.internal_count += 1
                    eid = f"e{self.pid+1}{self.internal_count}"
                    self.log(f"executes internal event (no neighbors) {eid}")
                else:
                    self.vc[self.pid] += 1
                    nei = self.rng.choice(self.adj[self.pid])
                    self.send_count += 1
                    mid = f"m{self.pid+1}{self.send_count}"
                    if self.algo == 'VC':
                        # send full vector along with message id
                        global_queues[nei].put((self.pid, mid, list(self.vc)))
                        self.sent_count += 1
                        self.log(f"sends message {mid} to Process{nei+1}, sent_entries={len(self.vc)}")
                    else:
                        # SK: Send only necessary entries based on changes and importance
                        changed = {}
                        self.current_time += 1
                        
                        # Always include the sender's own entry and any direct changes
                        changed = {self.pid: self.vc[self.pid]}  # Always send own entry
                        
                        # Include entries that have changed significantly
                        for i in range(self.n):
                            if i != self.pid and self.vc[i] > self.last_sent[nei][i]:
                                changed[i] = self.vc[i]
                                self.last_sent[nei][i] = self.vc[i]
                                self.last_update_time[i] = self.current_time
                        
                        # Include additional important entries up to n/2 total entries
                        remaining = max(self.n//2 - len(changed), 0)
                        if remaining > 0:
                            candidates = []
                            for i in range(self.n):
                                if i not in changed:
                                    age = self.current_time - self.last_update_time[i]
                                    importance = self.vc[i] * (age + 1)  # Age + 1 to consider non-zero current values
                                    if importance > 0:  # Only consider entries with some importance
                                        candidates.append((i, importance))
                            
                            # Add most important remaining entries
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            for i, _ in candidates[:remaining]:
                                changed[i] = self.vc[i]
                                self.last_sent[nei][i] = self.vc[i]
                                self.last_update_time[i] = self.current_time

                        global_queues[nei].put((self.pid, mid, changed))
                        self.sent_count += 1
                        self.log(f"sends optimized message {mid} to Process{nei+1}, sent_entries={len(changed)}")

        # mark finished
        # leave remaining messages for others to process
        return


def run_simulation(algorithm, inp_path):
    path = Path(inp_path)
    if not path.exists():
        print('Input file not found:', inp_path)
        return 1
    with path.open() as f:
        # Read parameters
        header = f.readline().strip().split()
        n = int(header[0]); lam = float(header[1]); alpha = float(header[2]); m = int(header[3])
        
        # Read topology
        adj = [[] for _ in range(n)]
        for i in range(n):
            line = f.readline().strip()
            if line:
                adj[i] = [int(x)-1 for x in line.split()]

    # Global queues representing each process inbox
    global global_queues
    global_queues = [Queue() for _ in range(n)]
    logs_per_proc = [[] for _ in range(n)]
    stop_event = threading.Event()

    procs = [Process(i, n, adj, lam, alpha, m, algorithm, logs_per_proc[i], stop_event) for i in range(n)]

    for p in procs:
        p.start()

    # wait until all processes have sent at least m messages
    try:
        while True:
            if all(p.sent_count >= m for p in procs):
                # give a short grace period for message delivery and processing
                time.sleep(0.5)
                stop_event.set()
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        stop_event.set()

    for p in procs:
        p.join()

    # write common logfile
    outname = f"common_log_{algorithm}.txt"
    # combine logs from all processes, parse ms prefix, sort by ms and write common logfile
    all_entries = []
    for i in range(n):
        for line in logs_per_proc[i]:
            line = line.strip()
            if not line: continue
            try:
                parts = line.split(' ', 1)
                ms = int(parts[0])
                text = parts[1]
                all_entries.append((ms, text))
            except Exception:
                # fallback: if malformed, use 0
                all_entries.append((0, line))
    all_entries.sort(key=lambda x: x[0])
    with open(outname, 'w', encoding='utf-8') as out:
        for ms, text in all_entries:
            out.write(text + '\n')
    print('Wrote', outname)
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python simulate_assignment.py [VC|SK] inp-params.txt')
        sys.exit(1)
    algo = sys.argv[1].upper()
    inp = sys.argv[2]
    if algo not in ('VC', 'SK'):
        print('Algorithm must be VC or SK')
        sys.exit(1)
    sys.exit(run_simulation(algo, inp))
