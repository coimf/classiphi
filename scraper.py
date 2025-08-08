import json
import os
import time
import logging
import re
from datetime import datetime
from typing_extensions import Literal, TypedDict, Pattern
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request

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
        "Referer": "https://www.google.com/",
    }
    try:
        req = Request(url=url,headers=headers)
        html = BeautifulSoup(urlopen(req), "html.parser")
    except Exception:
        failed_reqs += 1
        # :| friggin type checker
        return None
    return html

def scrape_ticks(base_url: str) -> list[str]:
    html = get_html(base_url)
    groups = html.find_all('div', class_='mw-category-group')
    links = []
    for group in groups:
        ul = group.find('ul')
        if ul:
            for li in ul.find_all('li'):
                a = li.find('a')
                links.append(a['href'])
    return links

def save_ticks_to_folder(topic: Literal['Algebra', 'Geometry', 'Number_Theory', 'Combinatorics'], data_folder: str) -> None:
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    with open("tick_urls.json", 'r') as f:
        URLS = json.load(f)

    total_ticks_scraped = 0
    for lv in ["intro", "inter", "olympiad", "trig"]:
        if lv == "trig" and topic.lower() != "geometry":
            continue
        if not os.path.exists(os.path.join(data_folder, f"{lv}_{topic.lower()}_ticks.json")):
            lv_ticks = []
            logging.info(f"Scraping {lv.capitalize()} {topic} ticks...")
            for url in tqdm(URLS[topic.lower()][lv]):
                lv_ticks += scrape_ticks(url)
            with open(os.path.join(data_folder, f"{lv}_{topic.lower()}_ticks.json"), 'w') as f:
                json.dump(lv_ticks, f)
            num_ticks = len(lv_ticks)
            total_ticks_scraped += num_ticks
            logging.info(f"Scraped {num_ticks} {lv.capitalize()} {topic} ticks.")

    if total_ticks_scraped > 0:
        logging.info(f"Scraped {total_ticks_scraped} total ticks to {data_folder}")

def scrape_problem(url: str) -> ProblemData:
    global total_problems_scraped

    def is_answer_choices(latex: str, num_choices_threshold: int = 4) -> bool:
        _bolded_label: Pattern = re.compile(r'\\text\s*\{\s*([A-Za-z])\.?\s*\}')
        _begin_choices: Pattern = re.compile(r'\\begin\{choices\}')
        if _begin_choices.search(latex):
            return True

        total_bolds = latex.count(r'\text')
        if total_bolds >= num_choices_threshold:
            return True

        labels = set(_bolded_label.findall(latex))
        if len(labels) >= num_choices_threshold:
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
        "title": "i got no idea man",
        "problem": "",
        "answer_choices": "",
        "answer": "idk",
        "solutions": ["bing", "bong", "ding", "dong"]
    }
    html = get_html(url)
    if not html:
        return problem_data

    title = html.find('h1', {'class': 'firstHeading', 'id': 'firstHeading'}).text.strip()
    problem_data['id'] = get_problem_id(title)
    problem_data['title'] = title

    problem_heading = html.find('span', {'class': 'mw-headline', 'id': 'Problem'})
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

    if not problem_heading:
        return problem_data

    paragraphs = []
    if problem_heading:
        problem_section = problem_heading.parent
        if problem_section:
            curr_paragraph = problem_section.next_sibling
            while curr_paragraph and curr_paragraph != solution_heading_span.parent:
                if hasattr(curr_paragraph, 'name') and curr_paragraph.name == 'p':
                    paragraphs.append(retrieve_latex_text(curr_paragraph))
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
                    curr_solution += retrieve_latex_text(curr_paragraph)
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
        problem_data['problem'] = paragraphs[0]
        problem_data['answer_choices'] = "frq"
        return problem_data

def retrieve_latex_text(paragraph) -> str:
    problem = []
    if not hasattr(paragraph, 'children'):
        #nolatex
        return str(paragraph)
    for element in paragraph.children:
        if hasattr(element, 'name') and element.name:
            if element.name == 'img':
                classes = element.get('class', [])
                if any('latex' in cls for cls in classes):
                    latex = element.get('alt', '')
                    if latex:
                        if 'latexcenter' in classes:
                            problem.append(f'\n\n{latex}\n\n')
                        else:
                            if problem and not problem[-1].endswith(' ') and not problem[-1].endswith('\n'):
                                problem.append(' ')
                            problem.append(latex+' ')
                else:
                    pass
            elif element.name == 'a':
                problem.append(element.text)
            else:
                if hasattr(element, 'children'):
                    problem.append(retrieve_latex_text(element))
                else:
                    problem.append(element.text if hasattr(element, 'text') else str(element))
        else:
            text = str(element)
            if text.strip():
                problem.append(text)
    final_prob = concat_parts(problem)
    final_prob = re.sub(r'[ \t]+', ' ', final_prob)
    final_prob = re.sub(r' *\n *', '\n', final_prob)
    return final_prob.strip()

def scrape_problems(base_url: str, ticks_filepath: str) -> dict[str, ProblemData]:
    global failed_problems_scraped
    if not os.path.exists(ticks_filepath) or not os.path.splitext(ticks_filepath)[1] == '.json':
        raise FileNotFoundError(f"JSON Ticks file not found at {ticks_filepath}")
    with open(ticks_filepath, 'r') as f:
        ticks = json.load(f)

    problems = {}
    logging.info(f"Scraping {ticks_filepath.removeprefix('scraped_data/ticks/').removesuffix('_ticks.json').replace('_', ' ')} problems...")
    for tick in tqdm(ticks):
        url = base_url + tick
        try:
            problem = scrape_problem(url)
            if problem is not None and problem['problem'].strip():
                problems[problem['title'].lower().replace(' ', '_')] = problem
            else:
                failed_problems_scraped += 1
        except Exception as e:
            logging.error(f"Failed to scrape problem at {url}: {e}")
            failed_problems_scraped += 1
    return problems

def save_problems_to_folder(topic: Literal['Algebra', 'Geometry', 'Combinatorics', 'Number_Theory'], data_folder: str) -> None:
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for lv in ["intro", "inter", "olympiad", "trig"]:
        if topic.lower() != "geometry" and lv == "trig":
            continue
        if not os.path.exists(os.path.join(data_folder, f'{lv}_{topic.lower()}_problems.json')):
            problems = scrape_problems('https://artofproblemsolving.com', f'scraped_data/ticks/{lv}_{topic.lower()}_ticks.json')
            with open(os.path.join(data_folder, f'{lv}_{topic.lower()}_problems.json'), 'w') as f:
                json.dump(problems, f)
            logging.info(f"Saved {len(problems)} {lv.capitalize()} {topic} problems.")

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

    #type checker sucks
    topics: list[Literal['Algebra', 'Geometry', 'Number_Theory', 'Combinatorics']] = ['Algebra', 'Geometry', 'Number_Theory', 'Combinatorics']
    for topic in topics:
        save_ticks_to_folder(topic, 'scraped_data/ticks/')
        save_problems_to_folder(topic, 'scraped_data/problems/')

    end = time.time()
    total_time = end-start
    total_hrs, total_mins, total_secs = int(total_time//3600), int((total_time%3600)//60), int(total_time%60)
    avg_secs_per_req = total_time/total_reqs if total_reqs else 0

    logging.info(f"""

        PERFORMANCE

        Began scraping at:        {time.strftime('%H:%M:%S', time.localtime(start))}
        Finished scraping at:     {time.strftime('%H:%M:%S', time.localtime(end))}
        Wall time:                {total_hrs:02d}:{total_mins:02d}:{total_secs:02d}

        Total requests made:      {total_reqs}
        Failed requests:          {failed_reqs} ({failed_reqs / total_reqs * 100:.2f}%)
        Average time per request: {avg_secs_per_req:.2f}s

        Total problems scraped:   {total_problems_scraped}
        Failed problems scraped:  {failed_problems_scraped} ({failed_problems_scraped / total_problems_scraped * 100:.2f}%)

    """)

if __name__ == '__main__':
    main()
