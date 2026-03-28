# %% setup
# ok so i'm just gonna load a few rows from advbench and see what the regex catches
# been meaning to do this for a week

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import matplotlib.pyplot as plt

advbench_path = Path(__file__).parent.parent / "conti" / "data" / "advbench_subset.jsonl"

# %% load rows
rows = []
with open(advbench_path) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"loaded {len(rows)} rows")
for r in rows[:5]:
    print(">>", r.get("prompt", ""))

# %% run regex verifier on them
#  these patterns are the same ones in verifier/keyword.py
# basically just copying them here bc importing the module was annoying
BLOCK_PATTERNS = [
    r"\b(bomb|weapon|explosive|grenade)\b",
    r"\b(how to make|how to build|how to create|instructions for)\b.{0,50}\b(poison|drug|meth|fentanyl|malware|ransomware)\b",
    r"\b(kill|murder|assassinate)\b.{0,40}\b(person|human|someone|president|politician)\b",
]

flags = re.IGNORECASE
compiled = [re.compile(p, flags) for p in BLOCK_PATTERNS]

results = {"caught": 0, "missed": 0}
for r in rows:
    p = r.get("prompt", "")
    caught = any(pat.search(p) for pat in compiled)
    if caught:
        results["caught"] += 1
    else:
        results["missed"] += 1
        # print(f"MISSED: {p[:80]}")  # uncomment this to see what slipped through

print(results)

# %% dummy plot for the presentation slide, run real eval later
labels = list(results.keys())
values = list(results.values())

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(labels, values, color=["steelblue", "tomato"])
ax.set_title("Verifier: regex catch rate on AdvBench subset")
ax.set_ylabel("count")
plt.tight_layout()
plt.savefig("scratchpad_plot.png", dpi=120)
plt.show()
print("plot saved, this is obviously not real eval just sanity checking")
