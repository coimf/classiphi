import json
import os
import time
import logging
import re
import browser_cookie3
from datetime import datetime
from typing_extensions import Literal, TypedDict, Pattern
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag, NavigableString
import requests

class ProblemID(TypedDict):
    year: int
    contest_name: str
    problem_number: int

class ProblemData(TypedDict):
    id: ProblemID
    title: str
    problem: str
    answer_choices: str
    answer: str
    solutions: list[str]

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[1;31m'
    }
    RESET = '\033[0m'

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S.")+f"{int(dt.microsecond / 1000):03d}"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        asctime = self.formatTime(record)
        prefix = f"{color}{asctime}: {record.levelname}{self.RESET}"
        message = record.getMessage()
        return f"{prefix}\n\t{message}"

def concat_parts(parts: list[str]) -> str:
    return ' '.join(parts).strip().replace('  ', ' ').replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')

def get_html(url: str) -> BeautifulSoup | None:
    global total_reqs, failed_reqs
    import random
    total_reqs += 1
    uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:116.0) Gecko/20100101 Firefox/116.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188",
        "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.92 Mobile Safari/537.36",
        "Mozilla/5.0 (Android 13; Mobile; rv:68.0) Gecko/68.0 Firefox/116.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.2420.81",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        "Mozilla/5.0 (X11; Linux i686; rv:84.0) Gecko/20100101 Firefox/84.0"
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15"
        "Mozilla/5.0 (Windows NT 10.0; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 Edg/87.0.664.66"
        "Mozilla/5.0 (X11; CrOS x86_64 13505.63.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Mobile Safari/537.36"
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        "Mozilla/5.0 (Linux; Android 10; SM‑A205U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.101 Mobile Safari/537.36"
    ]
    headers = {
        "User-Agent": random.choice(uas),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        #some idiot mispelled referrer
        "Referer": random.choice([
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://www.yahoo.com/",
            "https://www.reddit.com/",
            "https://www.twitter.com/",
            "https://www.instagram.com/",
            "https://www.facebook.com/",
            "https://www.linkedin.com/",
            "https://www.pinterest.com/",
            "https://www.tiktok.com/",
            "https://www.x.com/",
            "https://www.youtube.com/",
            "https://www.twitch.tv/",
        ])
    }
    #need aops sign in
    cj = browser_cookie3.chrome(domain_name="artofproblemsolving.com")
    try:
        r = requests.get(url, headers=headers, cookies=cj, timeout=15)
        r.raise_for_status()
        html = BeautifulSoup(r.text, "html.parser")
    except Exception:
        failed_reqs += 1
        return None
    return html

def scrape_p_index(base_url: str) -> list[str]:
    html = get_html(base_url)
    links = []

    if html and any(ts in base_url.lower() for ts in ['aime', 'amc_10', 'amc_12']):
        table = html.find_all('table', class_='wikitable')
        if len(table) > 0:
            tbody = table[-1].find('tbody')
            for tr in tbody.find_all('tr'):
                cells = tr.find_all(['th', 'td'])
                for c in cells:
                    if hasattr(c, 'a'):
                        a = c.find('a')
                        if a:
                            links.append("https://artofproblemsolving.com"+a['href'])
    elif html and 'amc_8' in base_url.lower():
        ul = html.find('ul')
        if ul:
            for li in ul.find_all('li'):
                a = li.find('a')
                if a:
                    links.append("https://artofproblemsolving.com"+a['href'])

    return links

def scrape_ticks(base_url: str) -> list[str]:
    html = get_html(base_url)
    links = []
    if not html:
        return []

    container = html.find('div', class_='mw-parser-output')
    main_list = container.find('ul')
    if not main_list:
        return []

    answer_key_li = None
    for li in main_list.find_all('li', recursive=False):
        a = li.find('a')
        if a and 'answer key' in a.text.strip().lower():
            answer_key_li = li
            break

    if not answer_key_li:
        return []

    target_list = answer_key_li.find('ul')
    if not target_list:
        target_list = container.find('ul')
        if not target_list:
            logging.info(f"No UL found on {base_url}, skipping")
            return []
        for i, li in enumerate(target_list.find_all('li')):
            if i < 2:
                continue
            a = li.find('a', href=True)
            if a and 'key' not in a.text.strip().lower():
                links.append("https://artofproblemsolving.com"+a['href'])
        return links


    for li in target_list.find_all('li'):
        a = li.find('a', href=True)
        if a:
            links.append("https://artofproblemsolving.com"+a['href'])

    return links

def save_ticks_to_folder(test: Literal['amc8', 'amc10', 'amc12', 'aime'], data_folder: str) -> None:
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    with open("amc_tick_urls.json", 'r') as f:
        URLS = json.load(f)

    total_ticks_scraped = 0
    if not os.path.exists(os.path.join(data_folder, f"{test.lower()}_url_ticks.json")):
        logging.info(f"Scraping {test.capitalize()} url ticks...")
        p_index_ticks = scrape_p_index(URLS[test][0])
        num_ticks = len(p_index_ticks)
        total_ticks_scraped += num_ticks
        with open(os.path.join(data_folder, f"{test.lower()}_url_ticks.json"), 'w') as f:
            json.dump(p_index_ticks, f)
        logging.info(f"Scraped {num_ticks} {test} ticks.")
        if total_ticks_scraped > 0:
            logging.info(f"Scraped {total_ticks_scraped} total url ticks to {data_folder}")

    if not os.path.exists(os.path.join(data_folder, f"{test.lower()}_problem_ticks.json")):
        logging.info(f"Scraping {test} problem ticks...")
        with open(os.path.join(data_folder, f"{test.lower()}_url_ticks.json"), 'r') as f:
            url_ticks = json.load(f)
        problem_ticks = []
        for test_url in tqdm(url_ticks):
            tk = scrape_ticks(test_url)
            while len(tk) == 0:
                tk = scrape_ticks(test_url)
            problem_ticks += tk
            if len(problem_ticks) > 0:
                with open(os.path.join(data_folder, f"{test.lower()}_problem_ticks.json"), 'w') as f:
                    json.dump(problem_ticks, f, indent=4)

def scrape_problem(url: str, max_retries: int = 3, max_wait_time_ms: float = 1728) -> ProblemData:
    global total_problems_scraped

    def is_answer_choices(latex: str, num_choices_threshold: int = 4) -> bool:
        exclude_envs = [
            r'\\begin\{(eqnarray\*?|array|cases|aligned|align\*?)\}.*?\\end\{\1\}',
            r'\[asy\].*?\[/asy\]'
        ]

        cleaned_latex = latex
        for pattern in exclude_envs:
            cleaned_latex = re.sub(pattern, '', cleaned_latex, flags=re.DOTALL)

        label_pattern = re.compile(
            r'(\\mathrm|\\text)?\s*\{?\(?([A-E])\)?\}?\s*'
        )

        matches = label_pattern.findall(cleaned_latex)

        candidate_labels = []
        for m in label_pattern.finditer(cleaned_latex):
            full_match = m.group(0)
            letter = m.group(2)
            if '(' in full_match and ')' in full_match:
                candidate_labels.append(letter)

        unique_labels = set(candidate_labels)
        if len(unique_labels) >= num_choices_threshold:
            return True

        if re.search(r'\\begin\{choices\}', latex):
            return True

        fallback_count = 0
        fallback_pattern = re.compile(r'(\\mathrm|\\text)\s*\{[^}]*\(?[A-E]\)?[^}]*\}')
        fallback_count = len(fallback_pattern.findall(latex))
        if fallback_count >= num_choices_threshold:
            return True

        return False

    def get_problem_id(title: str) -> ProblemID:
        year_match = re.match(r'^(\d{4}) ', title)
        if year_match:
            year = int(year_match.group(1))
            title_after_year = title[len(year_match.group(0)):]
        else:
            year = -1
            title_after_year = title

        problem_number_match = re.search(r'Problem[s]?/Problem (\d+)$', title)
        if problem_number_match:
            problem_number = int(problem_number_match.group(1))
        else:
            problem_number = -1

        contest_name = re.sub(r' Problems?/Problem \d+$', '', title_after_year)
        return {
            "year": year,
            "contest_name": contest_name,
            "problem_number": problem_number
        }

    def extract_answer(solution: str) -> str:
        results = []
        i = 0
        while True:
            #who invented fbox bruh why cant ye just use boxed????
            idx_boxed = solution.find('\\boxed{', i)
            idx_fbox = solution.find('\\fbox{', i)
            idx_candidates = [(idx_boxed, '\\boxed{'), (idx_fbox, '\\fbox{')]
            idx_candidates = [(idx, pat) for idx, pat in idx_candidates if idx != -1]
            if not idx_candidates:
                break

            idx, pat = min(idx_candidates, key=lambda x: x[0])
            start = idx + len(pat)
            brace_count = 1
            j = start
            while j < len(solution) and brace_count > 0:
                if solution[j] == '{':
                    brace_count += 1
                elif solution[j] == '}':
                    brace_count -= 1
                j += 1

            results.append(solution[start:j-1])
            i = j
        return results[-1] if results else ""

    problem_data: ProblemData = {
        "id": {
            "year": -1,
            "contest_name": "The Skibidi Contest",
            "problem_number": -1
        },
        "title": "who knows",
        "problem": "",
        "answer_choices": "",
        "answer": "idk",
        "solutions": ["we", "ran", "out", "of", "retries"]
    }

    html = get_html(url)
    if not html:
        import time
        for _ in range(max_retries):
            time.sleep(max_wait_time_ms/1000)
            html = get_html(url)
            if html:
                break
        return problem_data

    title = html.find('h1', {'class': 'firstHeading', 'id': 'firstHeading'}).text.strip()
    problem_data['id'] = get_problem_id(title)
    problem_data['title'] = title

    problem_heading_span = html.find('span', {'class': 'mw-headline', 'id': 'Problem'})
    solution_heading_span = html.find('span', {'class': 'mw-headline', 'id': 'Solution'})
    stop_heading = html.find('span', {'class': 'mw-headline', 'id': 'See_Also'})
    video_solution_heading = html.find('span', {
        'class': 'mw-headline',
        'id': re.compile(r'(?=.*video)(?=.*solution)', re.IGNORECASE)
    })
    if video_solution_heading:
        stop_heading = video_solution_heading
        stop_element_h2 = stop_heading.parent
    if not stop_heading:
        stop_element_h2 = html.find_all(
            'table',
            {'class': 'wikitable'}
        )[-1]
        if stop_element_h2:
            #end h2 must be before table
            stop_element_h2 = stop_element_h2.previous_sibling
    else:
        stop_element_h2 = stop_heading.parent
    if not solution_heading_span:
        #they rly gotta standardize their headings
        solution_heading_span = html.find('span', {
            'class': 'mw-headline',
            'id': re.compile(r'^Solution_[\w()]+')
        })
        if not solution_heading_span:
            solution_heading_span = stop_heading

    if not problem_heading_span:
        return problem_data

    paragraphs = []
    if problem_heading_span:
        problem_h2 = problem_heading_span.parent
        if problem_h2:
            curr_paragraph = problem_h2.next_sibling
            while curr_paragraph and curr_paragraph != solution_heading_span.parent:
                #aime 1984 q15 has a div for centered latex bruh
                #nooo 1990 aime q4 uses center!?
                if hasattr(curr_paragraph, 'name') and any(e == curr_paragraph.name for e in ['p', 'div', 'center']):
                    latex = retrieve_latex(curr_paragraph)
                    if latex:
                        paragraphs.append(latex)
                #2023 aime I 15 has a list
                elif hasattr(curr_paragraph, 'name') and curr_paragraph.name == 'ul':
                    for li in curr_paragraph.find_all('li'):
                        latex = retrieve_latex(li)
                        if latex:
                            paragraphs.append(latex)
                curr_paragraph = curr_paragraph.next_sibling

    solutions = []
    if solution_heading_span:
        solution_h2 = solution_heading_span.parent
        if solution_h2:
            curr_paragraph = solution_h2.next_sibling
            while curr_paragraph == "\n":
                curr_paragraph = curr_paragraph.next_sibling
            curr_solution = ""
            while curr_paragraph and curr_paragraph != stop_element_h2:
                if hasattr(curr_paragraph, 'name') and curr_paragraph.name == 'p':
                    curr_solution += retrieve_latex(curr_paragraph)
                elif hasattr(curr_paragraph, 'name') and curr_paragraph.name == 'h2':
                    #reset and skip h2 (multisolution heading)
                    if curr_solution:
                        solutions.append(curr_solution.replace('  ', ' ').replace(' ,', ',').replace(' .', '.').replace('\n\n\n', '\n\n'))
                    curr_solution = ""
                curr_paragraph = curr_paragraph.next_sibling
            if curr_solution:
                solutions.append(curr_solution.replace('  ', ' ').replace(' ,', ',').replace(' .', '.').replace('\n\n\n', '\n\n'))

    problem_data['solutions'] = solutions

    if problem_data['solutions'][0]:
        problem_data['answer'] = extract_answer(problem_data['solutions'][0])
    else:
        problem_data['answer'] = ""

    if not paragraphs:
        return problem_data
    if is_answer_choices(paragraphs[-1]):
        problem_data['problem'] = concat_parts(paragraphs[:-1])
        problem_data['answer_choices'] = paragraphs[-1].replace('  ', ' ').replace(' ,', ',').replace(' .', '.')
        return problem_data
    else:
        #likely FRQ problem
        problem_data['problem'] = concat_parts(paragraphs)
        problem_data['answer_choices'] = "frq"
        return problem_data

def _recursive_extractor(element) -> str:
    if isinstance(element, NavigableString):
        return str(element)

    if element.name in ['script', 'style']:
        return ''

    content_parts = []

    if element.name == 'img' and 'latex' in element.get('class', []):
        latex_code = element.get('alt', '')
        if latex_code:
            content_parts.append(f" {latex_code} ")

    if hasattr(element, 'children'):
        for child in element.children:
            content_parts.append(_recursive_extractor(child))

    return "".join(content_parts)

def retrieve_latex(element) -> str:
    raw_text = _recursive_extractor(element)
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    return cleaned_text.strip()

def retrieve_latex2(paragraph) -> str:
    problem = []
    if not hasattr(paragraph, 'children'):
        #nolatex
        return str(paragraph)
    for element in paragraph.children:
        print(element)
        if hasattr(element, 'name') and element.name:
            if element.name == 'img':
                classes = element.get('class', [])
                if any('latex' in cls for cls in classes):
                    latex = element.get('alt', '')
                    print(latex)
                    if latex:
                        if 'latexcenter' in classes:
                            problem.append(f'\n\n{latex}\n\n')
                        else:
                            if problem and not problem[-1].endswith(' ') and not problem[-1].endswith('\n'):
                                problem.append(' ')
                            problem.append(latex+' ')

                    #2025 amc8 q1
                    if element.text.strip():
                        problem.append(element.text.strip())

            elif element.name == 'a':
                problem.append(element.text.strip())
            else:
                if hasattr(element, 'children'):
                    problem.append(retrieve_latex(element))
                else:
                    problem.append(element.text if hasattr(element, 'text') else str(element))
        else:
            print(element)
            text = str(element)
            if text.strip():
                problem.append(text)
    final_prob = concat_parts(problem)
    final_prob = re.sub(r'[ \t]+', ' ', final_prob)
    final_prob = re.sub(r' *\n *', '\n', final_prob)
    return final_prob.strip()

def scrape_problems(base_url: str, ticks_filepath: str, max_retry_cycles: int = 3) -> dict[str, ProblemData]:
    global failed_problems_scraped
    if not os.path.exists(ticks_filepath) or not os.path.splitext(ticks_filepath)[1] == '.json':
        raise FileNotFoundError(f"JSON Ticks file not found at {ticks_filepath}")
    with open(ticks_filepath, 'r') as f:
        ticks = json.load(f)

    problems = {}
    ticks_to_retry = []

    logging.info(f"Scraping {ticks_filepath.removeprefix('scraped_data/ticks/').removesuffix('_problem_ticks.json').replace('_', ' ')} problems...")
    for tick in tqdm(ticks, desc="Scraping problems"):
        if 'redlink' in tick:
            continue
        try:
            problem = scrape_problem(tick)
            if problem is not None and problem['problem'].strip():
                problems[problem['title'].lower().replace(' ', '_')] = problem
            else:
                ticks_to_retry.append(tick)
        except Exception:
            ticks_to_retry.append(tick)

    for i in range(max_retry_cycles):
        if not ticks_to_retry:
            break
        logging.info(f"Failed to scrape {len(ticks_to_retry)} problems. Retrying... ({i+1}/{max_retry_cycles} retry cycles)")
        new_ticks_to_retry = []
        for tick in tqdm(ticks_to_retry, desc=f"Retry cycle {i+1}"):
            url = base_url + tick
            try:
                problem = scrape_problem(url)
                if problem is not None and problem['problem'].strip():
                    key = problem['title'].lower().replace(' ', '_')
                    problems[key] = problem
                else:
                    if i < max_retry_cycles - 1:
                        new_ticks_to_retry.append(tick)
                    else:
                        failed_problems_scraped += 1
            except Exception as e:
                if i < max_retry_cycles - 1:
                    new_ticks_to_retry.append(tick)
                else:
                    failed_problems_scraped += 1
        ticks_to_retry = new_ticks_to_retry

    return problems

def save_problems_to_folder(test: Literal['amc8', 'amc10', 'amc12', 'aime'], data_folder: str) -> None:
    if not os.path.exists(os.path.join(data_folder, f'{test}_problems.json')):
        problems = scrape_problems('', f'scraped_data/ticks/{test}_problem_ticks.json')
        with open(os.path.join(data_folder, f'{test}_problems.json'), 'w') as f:
            json.dump(problems, f)
        logging.info(f"Saved {len(problems)} {test} problems.")

def run_test_cases(testcases_file: str, output_file: str = "test_results.json", run_first_n: int = 10) -> None:
    try:
        with open(testcases_file, 'r') as f:
            ticks = json.load(f)
    except FileNotFoundError:
        logging.error(f"No test cases file found at '{testcases_file}'.")
        quit()

    assert isinstance(run_first_n, int) and run_first_n > 0, "run_first_n must be a positive integer"

    scraped = {}
    ticks = ticks[:run_first_n]
    logging.info(f"Scraping {len(ticks)} suspicious problems...")
    for tick in tqdm(ticks):
        try:
            scraped[tick] = scrape_problem(tick)
        except Exception as e:
            logging.error(f"🚨🚨🚨 This problem is very suspicious! {tick}: {e}")

    with open(output_file, 'w') as f:
        json.dump(scraped, f)

    logging.info(f"Finished scraping suspicious problems. Check {output_file} for results.")
    quit()

def main() -> None:
    global total_reqs, total_problems_scraped, failed_problems_scraped, failed_reqs
    start = time.time()

    handler = logging.StreamHandler()
    formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S.%f")
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    total_reqs = 0
    failed_reqs = 0
    total_problems_scraped = 0
    failed_problems_scraped = 0

    # run_test_cases('failed.json')

    #type checker sucks
    tests: list[Literal['amc8', 'amc10', 'amc12', 'aime']] = ['amc8', 'amc10', 'amc12', 'aime']
    for test in tests:
        save_ticks_to_folder(test, 'scraped_data/ticks/')
        save_problems_to_folder(test, 'scraped_data/problems/')

    end = time.time()
    total_time = end-start
    total_hrs, total_mins, total_secs = int(total_time//3600), int((total_time%3600)//60), int(total_time%60)
    avg_secs_per_req = total_time/total_reqs if total_reqs else 0
    if total_reqs:
        failed_reqs_percent = failed_reqs / total_reqs * 100
    else:
        failed_reqs_percent = 0
    if total_problems_scraped:
        failed_problems_scraped_percent = failed_problems_scraped / total_problems_scraped * 100
    else:
        failed_problems_scraped_percent = 0

    logging.info(f"""

        PERFORMANCE

        Began scraping at:        {time.strftime('%H:%M:%S', time.localtime(start))}
        Finished scraping at:     {time.strftime('%H:%M:%S', time.localtime(end))}
        Wall time:                {total_hrs:02d}:{total_mins:02d}:{total_secs:02d}

        Total requests made:      {total_reqs}
        Failed requests:          {failed_reqs} ({failed_reqs_percent:.2f}%)
        Average time per request: {avg_secs_per_req:.2f}s

        Total problems scraped:   {total_problems_scraped}
        Failed problems scraped:  {failed_problems_scraped} ({failed_problems_scraped_percent:.2f}%)

    """)

if __name__ == '__main__':
    main()
