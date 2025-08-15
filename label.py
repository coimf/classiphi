import lmstudio as lms
import json
import os
import time
from typing import Literal
from tqdm import tqdm
from groq import Groq, RateLimitError
from google import genai
from cerebras.cloud.sdk import Cerebras

def label_with_llm(input_data: str, providers: dict[Literal['google', 'lmstudio', 'groq', 'cerebras'], dict[str, dict]], keys: dict[str, bool]) -> dict[str, str] | None:
    global key_no
    with open("response_schema.json", "r") as f:
        schema = json.load(f)
    with open("sys_prompt.txt", "r") as f:
        sys_prompt = f.read()
    for provider, models in providers.items():
        if provider == 'lmstudio':
            for model_id, config in models.items():
                try:
                    model = lms.llm(model_id, config={"contextLength": 4096})
                    chat = lms.Chat(sys_prompt)
                    chat.add_user_message(input_data)
                    result: lms.PredictionResult = model.respond(
                        chat,
                        config=config,
                        response_format=schema
                    )
                    data = json.loads(json.dumps(result.parsed))
                    data['annotator'] = model_id
                    data['reviewed'] = False
                    return data
                except Exception as e:
                    print(f"Failed to use {provider} model '{model_id}': {e}")
                    break
        elif provider == 'groq':
            if not any(list(keys.values())):
                keys = {key: True for key in keys}
            if not keys[f"GROQ_API_KEY{key_no}"]:
                continue
            client = Groq(
                api_key=os.environ.get(f"GROQ_API_KEY{key_no}")
            )
            for model_id, config in models.items():
                try:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": input_data}
                        ],
                        **config,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "CompetitionMathClassification",
                                "schema": schema
                            }
                        }
                    )
                    data = json.loads(completion.choices[0].message.content)
                    if isinstance(data, dict):
                        data['annotator'] = model_id
                        data['reviewed'] = False
                        return data
                    else:
                        print("Expected dictionary but got:", type(data))
                        return None
                except RateLimitError as e:
                    if "rate limit" in str(e).lower():
                        print(f"GROQ_API_KEY{key_no}: Rate limit hit for {model_id}: {e}")
                        keys[f"GROQ_API_KEY{key_no}"] = False
                        key_no += 1
                        key_no %= len(keys)
                        time.sleep(10)
                        continue
                    else:
                        print(f"Groq API error for {model_id}: {e}")
                        continue
                except Exception as e:
                    print(f"Unexpected error with {model_id}: {e}")
                    continue
        elif provider == 'google':
            client = genai.Client()
            for model_id, model_config in models.items():
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=sys_prompt+' '+input_data,
                        config={
                            "response_mime_type": "application/json",
                            "response_json_schema": schema
                        },
                    )
                    #json dict
                    data = response.parsed
                    if isinstance(data, dict):
                        data['annotator'] = model_id
                        data['reviewed'] = False
                        return data
                    else:
                        try:
                            data = json.loads(str(response.text))
                        except json.JSONDecodeError:
                            print(f"Google: JSON decode error for {model_id}: {response.text}.")
                            return None
                except RateLimitError as e:
                    if "rate limit" in str(e).lower():
                        print(f"Google: Rate limit hit for {model_id}.")
                        break
                    else:
                        print(f"Google API error for {model_id}: {e}")
                        continue
                except Exception as e:
                    print(f"Unexpected error with {model_id}: {e}")
                    break
        elif provider == 'cerebras':
            client = Cerebras(
                api_key=os.environ.get(f"CEREBRAS_API_KEY{key_no%2}"),
            )
            for model_id, config in models.items():
                try:
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": input_data}
                        ],
                        **config,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "CompetitionMathClassification",
                                "schema": schema
                            }
                        }
                    )
                    data = json.loads(completion.choices[0].message.content)
                    if isinstance(data, dict):
                        data['annotator'] = model_id
                        data['reviewed'] = False
                        return data
                    else:
                        print("Expected dictionary but got:", type(data))
                        return None
                except RateLimitError as e:
                    if "rate limit" in str(e).lower():
                        print(f"CEREBRAS_API_KEY{key_no}: Rate limit hit for {model_id}: {e}")
                        keys[f"CEREBRAS_API_KEY{key_no}"] = False
                        key_no += 1
                        key_no %= 2
                        continue
                    else:
                        print(f"Cerebras API error for {model_id}: {e}")
                        continue
                except Exception as e:
                    print(f"Unexpected error with {model_id}: {e}")
                    continue
    return None

def main() -> None:
    global key_no
    providers: dict[Literal['google', 'groq', 'lmstudio', 'cerebras'], dict[str, dict]] = {
        'groq': {
            'openai/gpt-oss-120b': {
                'temperature': 1,
                'max_completion_tokens': 8192,
                'top_p': 1,
                'reasoning_effort': "high",
                'stream': False
            },
            'moonshotai/kimi-k2-instruct': {
                'temperature': 0.6,
                'max_completion_tokens': 16384,
                'top_p': 1,
                'stream': False
            },
            'openai/gpt-oss-20b': {
                'temperature': 1,
                'max_completion_tokens': 8192,
                'top_p': 1,
                'reasoning_effort': "high",
                'stream': False
            },
        },
        'cerebras': {
            'gpt-oss-120b': {
                'temperature': 1,
                'max_completion_tokens': 65536,
                'top_p': 1,
                'reasoning_effort': "high",
                'stream': False
            },
            'qwen-3-235b-a22b-thinking-2507': {
                'max_completion_tokens': 65536,
                'stream': False
            },
            'qwen-3-235b-a22b-instruct-2507': {
                'max_completion_tokens': 65536,
                'stream': False
            },
            'qwen-3-32b': {
                'max_completion_tokens': 65536,
                'stream': False
            },
        },
    }

    key_no = 0
    groq_keys = {
        "GROQ_API_KEY3": True,
        "GROQ_API_KEY2": True,
        "GROQ_API_KEY1": True,
        "GROQ_API_KEY0": True,
    }

    tests = ["amc8", "amc10", "amc12", "aime"]
    for test in tests:
        if os.path.exists(f"scraped_data/problems/labeled_{test}_problems.json"):
            with open(f"scraped_data/problems/labeled_{test}_problems.json", "r") as f:
                problems = json.load(f)
        else:
            with open(f"scraped_data/problems/{test}_problems.json", "r") as f:
                problems = json.load(f)

        for id, problem in tqdm(problems.items(), desc=f"Labeling {test.upper()} problems"):
            if 'label' in list(problem.keys()):
                continue
            output_label_dict = label_with_llm(f"{problem['problem']} {' '.join(problem['solutions'][:3])}", providers, groq_keys)
            if not output_label_dict:
                print(f"Failed to label problem: {problem['problem']}")
                continue
            problems[id]['label'] = output_label_dict
            with open(f"scraped_data/problems/labeled_{test}_problems.json", "w") as f:
                json.dump(problems, f, indent=4)


if __name__ == "__main__":
    main()
