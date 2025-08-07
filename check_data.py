import json
import os
from transformers import BertTokenizer

def extract_topic(filename: str):
    #filename format: level_topic_problems.json
    parts = filename.split("_")
    if len(parts) >= 3:
        return parts[1]
    return ""

def main():
    total_problems = 0
    total_chars = 0
    total_tokens = 0
    min_chars = float('inf')
    max_chars = float('-inf')
    min_tokens = float('inf')
    max_tokens = float('-inf')
    tokenizer = BertTokenizer.from_pretrained('models/topic_classifier_9900_epoch3_0805_23-10-17')

    problem_files = os.listdir("scraped_data/problems")
    problem_files = sorted(problem_files, key=extract_topic)
    for file in problem_files:
        with open(f"scraped_data/problems/{file}", "r") as f:
            data = json.load(f)
            num_problems = len(data)
            print(f"{file:<40} {num_problems:>5} problems")
            total_problems += num_problems

            for problem in data:
                char_len = len(problem)
                tokens = tokenizer.tokenize(problem)
                token_len = len(tokens)

                total_chars += char_len
                total_tokens += token_len

                min_chars = min(min_chars, char_len)
                max_chars = max(max_chars, char_len)
                min_tokens = min(min_tokens, token_len)
                max_tokens = max(max_tokens, token_len)
    total_ticks = 0
    problem_files = os.listdir("scraped_data/ticks")
    for file in sorted(problem_files):
        with open(f"scraped_data/ticks/{file}", "r") as f:
            data = json.load(f)
            num_ticks = len(data)
            total_ticks += num_ticks

    print(f"...\n{'Total problems listed on wiki:':<40} {total_ticks:>5}")
    print(f"{'Total problems scraped:':<40} {total_problems:>5} ({100 * total_problems / total_ticks:.2f}%)")
    print(f"{'Total characters:':<40} {total_chars:>8}")
    print(f"{'Total tokens:':<40} {total_tokens:>8}")
    print(f"{'Average sequence length:':<40} {total_tokens / total_problems:>8.2f} tokens")
    print(f"{'Shortest sequence length:':<40} {min_tokens:>8} tokens")
    print(f"{'':<40} {min_chars:>8} chars")
    print(f"{'Longest sequence length:':<40} {max_tokens:>8} tokens")
    print(f"{'':<40} {max_chars:>8} chars")
    print(f"{'Vocab size:':<40} {len(tokenizer.vocab):>8}")

if __name__ == "__main__":
    main()
