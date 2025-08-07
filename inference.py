import streamlit as st
st.set_page_config(page_title="ClassiPhi | Classify Math Problems", page_icon="👾")
st.title("Math Problem Classifier")
with st.spinner("Loading models...", show_time=True):
    import pandas as pd
    import altair as alt
    import os
    from psutil import Process
    from random import sample
    from torch.nn.functional import softmax
    from torch import inference_mode, float16
    from transformers import BertForSequenceClassification, BertTokenizer
    from streamlit_extras.stylable_container import stylable_container
    from typing import Optional, Union, Tuple, Dict, cast
    from huggingface_hub import snapshot_download
    import history

@st.cache_resource
def load_model(model_name: str) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    if not os.path.exists(model_name):
        hf_token = st.secrets["HF_TOKEN"]
        with st.spinner("Downloading models...", show_time=True):
            print(f"Local models not detected. Downloading {model_name}")
            snapshot_download(
                repo_id="cof139/bert-classiphi",
                repo_type="model",
                local_dir="models/",
                allow_patterns=model_name.removeprefix("models/")+"/*",
                token=hf_token
            )
    model = BertForSequenceClassification.from_pretrained(model_name, torch_dtype=float16)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name, torch_dtype=float16)
    return model, tokenizer

@st.cache_data
def build_altair_bar_chart(df: pd.DataFrame, title: str) -> alt.Chart:
    max_idx = df['probability'].idxmax()
    df2 = df.reset_index().assign(
        label = lambda d: d['index'],
        color = lambda d: ['highlight' if i == max_idx else 'normal' for i in d['index']]
    )
    color_scale = alt.Scale(
        domain=['normal', 'highlight'],
        range=['gray', '#cb785c']
    )
    return (
        alt.Chart(df2)
        .mark_bar()
        .encode(
            y=alt.Y('label:N', sort='-x', title=None),
            x=alt.X('probability:Q', title='Probability'),
            color=alt.Color('color:N', scale=color_scale, legend=None)
        )
        .properties(title=title)
    )

@st.cache_data(show_spinner=False)
def classify_problem(
    problem: str,
    topic: Optional[str] = None,
    return_probabilities: bool = False
) -> Union[str, Tuple[str, Dict[str, float]]]:
    """
    Classifies either topic or skill of the given problem using respective classifier model.

    Args:
        problem (str): Text of problem to classify.
        topic (Optional[str]): The topic of the problem (if classifying skill).
        return_probabilities (bool): If True, also return normalized probabilities for all labels.

    Returns:
        str or (str, dict): Predicted label, optionally with dict of label probabilities.
    """
    classifier_name = (topic or 'topic').lower() + "_classifier"
    model, tokenizer = load_model(models[classifier_name]["model_path"])
    labels = models[classifier_name]['labels']

    encoded = tokenizer(problem.replace('$', ''), return_tensors="pt", padding=True, truncation=True)

    with inference_mode():
        logits = model(**encoded).logits

    predicted_id = int(logits.argmax(dim=-1).item())
    predicted_label = labels[predicted_id].lower()

    if not return_probabilities:
        return predicted_label

    probs = softmax(logits, dim=-1).half()[0]
    prob_vals = probs.cpu().numpy()
    probabilities_dict = {labels[i].lower(): float(prob_vals[i]) for i in range(len(labels))}
    del probs, logits, encoded
    return predicted_label, probabilities_dict

def load_streamlit_ui() -> None:
    st.subheader("Examples")
    if "shuffled_examples" not in st.session_state:
        st.session_state.shuffled_examples = sample(example_problems, len(example_problems))
    example_columns = st.columns(3, vertical_alignment="center")
    if "selected_example_problem" not in st.session_state:
        st.session_state.selected_example_problem = ""
    for i, column in enumerate(example_columns):
        example = st.session_state.shuffled_examples[i]
        if column.button(f"{example[:42]}...", key=f"example_btn_{i}"):  # Show a preview
            st.session_state.selected_example_problem = example
    problem = st.text_area(
        "Enter a problem:",
        placeholder="Paste a problem here...",
        value=st.session_state.selected_example_problem,
        height="content",
        label_visibility="hidden"
    )
    with stylable_container(key="classify", css_styles=r"""
        toggle { float: left; }
        button { float: right; }
    """):
        cols = st.columns(2, vertical_alignment="center")
        cols[0].badge("BETA", color="primary")
        classify_skill = cols[0].toggle("Classify Skill :material/experiment:", value=True)
        start_classification = cols[1].button("Classify", type="primary", disabled=(len(problem) == 0))
        st.divider()

    if start_classification and problem:
        with st.spinner("Classifying...", show_time=True):
            predicted_topic, topic_probabilities = cast(Tuple[str, Dict[str, float]], classify_problem(problem, return_probabilities=True))
            if classify_skill:
                predicted_skill, skill_probabilities = cast(Tuple[str, Dict[str, float]], classify_problem(problem, topic=predicted_topic, return_probabilities=True))
            else:
                predicted_skill = "disabled_by_user"
                skill_probabilities = None
        problem_section, predictions_section = st.columns([65, 35], border=True)
        st.divider()
        problem_section.markdown(problem)
        process = Process(os.getpid())
        predictions_section.code(f"{predicted_topic}", language=None, wrap_lines=True, height="stretch")
        if classify_skill and predicted_skill:
            predictions_section.code(f"{predicted_skill}", language=None, wrap_lines=True, height="stretch")
        predictions_section.code(f"Memory Usage\n{process.memory_info().rss / 1024 ** 2:.2f} MB")
        problem_id = history.add_history(
            problem,
            predicted_topic,
            predicted_skill,
            models['topic_classifier']['model_path'],
            models[f'{predicted_topic}_classifier']['model_path'],
            process.memory_info().rss / 1024 ** 2
        )
        topic_chart, skill_chart = st.columns([40 if classify_skill else 100, 60 if classify_skill else 1])
        topic_chart_altair = build_altair_bar_chart(pd.DataFrame.from_dict(topic_probabilities, orient='index').rename(columns={0: 'probability'}), "Topic Probabilities")
        topic_chart.altair_chart(topic_chart_altair, use_container_width=True)
        if skill_probabilities and classify_skill:
            skill_chart_altair = build_altair_bar_chart(pd.DataFrame.from_dict(skill_probabilities, orient='index').rename(columns={0: 'probability'}), "Skill Probabilities")
            skill_chart.altair_chart(skill_chart_altair, use_container_width=True)
        st.divider()
        request_feedback(problem_id)

@st.fragment
def request_feedback(id: str) -> None:
    st.write("Was this helpful?")
    selected = st.feedback("thumbs")
    if selected in (0, 1):
        st.success("Thank you for your feedback!")
        history.save_feedback(id, selected)

def main():
    global example_problems, models
    models = {
        "topic_classifier": {
            "model_path": "models/topic_classifier_9900_epoch3_0805_23-10-17",
            "labels": {
                0: "algebra",
                1: "geometry",
                2: "number_theory",
                3: "combinatorics"
            }
        },
        "algebra_classifier": {
            "model_path": "models/algebra_classifier_8158_epoch12_0729_21-45-15",
            "labels" : {
                0: "Simon's Favorite Factoring Trick",
                1: "Clever Algebraic Manipulations",
                2: "Advanced Systems of Equations",
                3: "Functional Operations",
                4: "Difference-of-Squares",
                5: "Rate Problems",
                6: "Arithmetic Series",
                7: "Absolute Value",
                8: "Geometric Series",
                9: "Quadratic Inequalities"
            }
        },
        "geometry_classifier": {
            "model_path": "models/geometry_classifier_8435_epoch6_0729_20-40-41",
            "labels" : {
                0: "Similar Triangles",
                1: "Bisectors in a Triangle",
                2: "Funky Circle Areas",
                3: "Special Right Triangles",
                4: "Inequalities in Triangles",
                5: "Isosceles and Equilateral Triangles",
                6: "Quadrilaterals",
                7: "Spheres",
                8: "Cones",
                9: "Transformations"
            }
        },
        "number_theory_classifier": {
            "model_path": "models/number_theory_classifier_7109_epoch6_0729_20-34-55",
            "labels" : {
                0: "The Last Digit (Base 10)",
                1: "Modular Arithmetic",
                2: "Remainders",
                3: "Greatest Common Divisor",
                4: "LCM and GCD",
                5: "Counting Divisors",
                6: "Prime Factorization",
                7: "Converting to Base 10",
                8: "Base Number Problem-Solving",
                9: "Repeating Decimals"
            }
        },
        "combinatorics_classifier": {
            "model_path": "models/combinatorics_classifier_7368_epoch16_0729_22-36-57",
            "labels": {
                0: "Constructive Counting",
                1: "Complementary Counting",
                2: "Casework Counting",
                3: "Counting Independent Events",
                4: "Advanced Probability with Combinations",
                5: "Geometric Probability",
                6: "Counting with Restrictions",
                7: "Complementary Probability",
                8: "Expected Value",
                9: "Counting with Symmetry"
            }
        }
    }
    example_problems = [
        r"""Let $p$, $q$, and $r$ be the distinct roots of the polynomial $x^3 - 22x^2 + 80x - 67$. It is given that there exist real numbers $A$, $B$, and $C$ such that $$\dfrac{1}{s^3 - 22s^2 + 80s - 67} = \dfrac{A}{s-p} + \dfrac{B}{s-q} + \dfrac{C}{s-r}$$for all $s\not\in\{p,q,r\}$. What is $\tfrac1A+\tfrac1B+\tfrac1C$?""",
        r"""A square with side length $x$ is inscribed in a right triangle with sides of length $3$, $4$, and $5$ so that one vertex of the square coincides with the right-angle vertex of the triangle. A square with side length $y$ is inscribed in another right triangle with sides of length $3$, $4$, and $5$ so that one side of the square lies on the hypotenuse of the triangle. What is $\dfrac{x}{y}$?""",
        r"""For how many integer values of $x$ is $|2x| \leq 7 \pi$?""",
        r"""For each positive integer $n$, let $f(n) = \sum_{k = 1}^{100} \lfloor \log_{10} (kn) \rfloor$. Find the largest value of $n$ for which $f(n) \le 300$.""",
        r"""Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$, inclusive, such that $12x^2-xy-6y^2=0$.""",
        r"""Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$""",
        r"""There are exactly $K$ positive integers $5 \leq b \leq 2024$ such that the base-$b$ integer $2024_{b}$ is divisible by $16$(where $16$ is in base ten). What is the sum of the digits of $K$?""",
        r"""Integers $a$, $b$, and $c$ satisfy $ab + c = 100$, $bc + a = 87$, and $ca + b = 60$. What is $ab + bc + ca?$""",
        r"""What is the remainder when $7^{2024}+7^{2025}+7^{2026}$ is divided by $19$?""",
        r"""Jerry likes to play with numbers. One day, he wrote all the integers from $1$ to $2024$ on the whiteboard. Then he repeatedly chose four numbers on the whiteboard, erased them, and replaced them by either their sum or their product. (For example, Jerry's first step might have been to erase $1$, $2$, $3$, and $5$, and then write either $11$, their sum, or $30$, their product, on the whiteboard.) After repeatedly performing this operation, Jerry noticed that all the remaining numbers on the whiteboard were odd. What is the maximum possible number of integers on the whiteboard at that time?""",
        r"""How many different remainders can result when the $100$th power of an integer is divided by $125$?""",
        r"""A rectangle has integer side lengths and an area of $2024$. What is the least possible perimeter of the rectangle?""",
        r"""Real numbers $a, b,$ and $c$ have arithmetic mean $0$. The arithmetic mean of $a^2, b^2,$ and $c^2$ is $10$. What is the arithmetic mean of $ab, ac,$ and $bc$?""",
        r"""A dartboard is the region $B$ in the coordinate plane consisting of points $(x,y)$ such that $|x| + |y| \le 8$ . A target $T$ is the region where $(x^2 + y^2 - 25)^2 \le 49.$ A dart is thrown and lands at a random point in $B$. The probability that the dart lands in $T$ can be expressed as $\frac{m}{n} \cdot \pi,$ where $m$ and $n$ are relatively prime positive integers. What is $m + n?$""",
        r"""A list of $9$ real numbers consists of $1$, $2.2$, $3.2$, $5.2$, $6.2$, and $7$, as well as $x$, $y$ , and $z$ with $x$ $\le$ $y$ $\le$ $z$. The range of the list is $7$, and the mean and the median are both positive integers. How many ordered triples ($x$, $y$, $z$) are possible?"""
    ]
    load_streamlit_ui()

if __name__ == "__main__":
    main()
