import json
import os

def main():
    total_problems = 0
    problem_files = os.listdir("scraped_data/problems")
    for file in sorted(problem_files):
        with open(f"scraped_data/problems/{file}", "r") as f:
            data = json.load(f)
            num_problems = len(data)
            print(f"{file:<40} {num_problems:>5} problems")
            total_problems += num_problems

    total_ticks = 0
    problem_files = os.listdir("scraped_data/ticks")
    for file in sorted(problem_files):
        with open(f"scraped_data/ticks/{file}", "r") as f:
            data = json.load(f)
            num_ticks = len(data)
            total_ticks += num_ticks

    print(f"...\n{'Total problems listed on wiki:':<40} {total_ticks:>5}")
    print(f"{'Total problems scraped:':<40} {total_problems:>5} ({100 * total_problems / total_ticks:.2f}%)")

if __name__ == "__main__":
    main()
