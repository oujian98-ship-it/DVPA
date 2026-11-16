"""Read-only concise status of real P0 runs, including interrupted ones."""
import csv
import json
from pathlib import Path

for directory in sorted(Path('results').glob('p0_*')):
    metrics=directory/'rounds.csv'
    if not metrics.exists():
        if (directory/'failure.json').exists():
            print(directory.name, 'failed before first release')
        else:
            print(directory.name, 'initializing')
        continue
    rows=list(csv.DictReader(metrics.open()))
    if not rows:
        continue
    last=rows[-1]
    status='failed' if (directory/'failure.json').exists() else ('completed' if (directory/'summary.json').exists() else 'in progress or interrupted')
    print(f"{directory.name}: {status}, round {last['round']}, accuracy {float(last['test_acc']):.2f}%, loss {float(last['test_loss']):.4g}, epsilon_max {float(last['epsilon_max']):.4f}")
