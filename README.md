<div align="center">

#  Classiphi

<div>
    Classifying competitive math problems by their topics and skills
</div>
<br>
</div>

## Online Demo

[https://classiphi.streamlit.app](https://classiphi.streamlit.app)

## Local Installation

Clone the repository:

```bash
git clone https://github.com/coimf/classiphi.git
cd classiphi
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Models

Classiphi uses five BERT models fine-tuned on competitive math problem classification. Models are available on [Hugging Face](https://huggingface.co/cof139/bert-classiphi). Original model is available [here](https://huggingface.co/aieng-lab/math_pretrained_bert_mamut).

## Architecture

A topic classifier model first determines the topic of a problem (out of four possible topics), and then the respective skill classifier model determines the skill of the problem (out of ten possible skills). The skill classifier model may produce inaccuracies.

## Data Collection

To scrape math problems from the [Art of Problem Solving](https://artofproblemsolving.com) wiki, run:

```bash
python scraper.py
```

Scraped problems are located at `scraped_data/problems`.

## Limitations

- Models may make incorrect classifications.
- Skill classifier models cannot output skills not present in the training data.
