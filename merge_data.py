import json
import os

all = []
tests = ["amc8", "amc10", "amc12", "aime"]
for test in tests:
    if os.path.exists(f"scraped_data/problems/labeled_{test}_problems.json"):
        with open(f"scraped_data/problems/labeled_{test}_problems.json", "r") as f:
            problems = json.load(f)
    for problem in problems.values():
        for key, value in problem['label'].items():
            problem[key] = value
        problem.pop('label', None)
        all.append(problem)
with open("merged_data_labeled.json", "w") as f:
    json.dump(all, f)
