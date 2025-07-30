import streamlit as st
st.set_page_config(page_title="Math Problem Classifier", page_icon="👾")
st.title("Math Problem Topic and Skill Classifier")
with st.spinner("Loading models...", show_time=True):
    import time
    import psutil
    import torch
    import pandas as pd
    import altair as alt
    from transformers import BertForSequenceClassification, BertTokenizer
    from streamlit_extras.stylable_container import stylable_container
    from typing import Optional, Union, Tuple, Dict

def load_models(models, device: str) -> None:
    """
    Loads models onto the specified device.

    Args:
        models (dict): Dictionary of models to load.
        device (str): Device to load models onto.

    Returns:
        None
    """
    device = device.lower()
    if device not in ['cpu', 'cuda', 'mps']:
        print(f"Device '{device}' not supported, defaulting to cpu.")
        device = "cpu"
    for model_name in models:
        model = models[model_name]['model']
        model = model.to(torch.device(device))
        model.eval()

def build_colored_bar_chart(df, title):
    df = df.copy()
    df['label'] = df.index
    max_index = df['probability'].idxmax()
    df['color'] = ['highlight' if idx == max_index else 'normal' for idx in df.index]

    color_scale = alt.Scale(
        domain=['normal', 'highlight'],
        range=['gray', '#cb785c']
    )

    chart = alt.Chart(df.reset_index()).mark_bar().encode(
        y=alt.Y('label:N', sort='-x', title=None),
        x=alt.X('probability:Q', title='Probability'),
        color=alt.Color('color:N', scale=color_scale, legend=None)
    ).properties(
        title=title
    )

    return chart

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
    global models, device
    classifier_name = f"{topic or 'topic'}_classifier".lower()
    tokenizer = models[classifier_name]['tokenizer']
    model = models[classifier_name]['model']
    labels = models[classifier_name]['labels']

    input_ids = tokenizer(problem, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_ids)
    logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0) #shape: (num_labels,)
    predicted_id = int(torch.argmax(probs).item())
    predicted_label = str(labels[predicted_id]).lower()

    if return_probabilities:
        probabilities_dict = {str(label).lower(): prob.item() for label, prob in zip(labels.values(), probs)}
        return predicted_label, probabilities_dict
    else:
        return predicted_label

def load_streamlit_ui():
    problem = st.text_area("# Enter a problem for classification:", placeholder="Paste a problem here...", height="content", label_visibility="hidden")
    with stylable_container(key="classify", css_styles=r"""
        button {
            float: right;
            margin-bottom: 20px;
        }
    """):
        start_classification = st.button("Classify", type="primary", disabled=(len(problem) == 0))

    if start_classification and problem:
        with st.spinner("Classifying...", show_time=True):
            predicted_topic = classify_problem(problem, return_probabilities=True)
            predicted_skill = classify_problem(problem, topic=predicted_topic[0], return_probabilities=True)

        problem_section, predictions_section = st.columns([5,2], border=True)
        st.divider()
        problem_section.markdown(problem)
        predictions_section.code(f"{predicted_topic[0]}", language=None, wrap_lines=True, height="stretch")
        predictions_section.code(f"{predicted_skill[0]}", language=None, wrap_lines=True, height="stretch")

        topic_probability_chart_data = pd.DataFrame.from_dict(
            dict(sorted(predicted_topic[1].items(), key=lambda item: item[1], reverse=True)), orient='index', columns=['probability']
        ).sort_values(by='probability', ascending=True)
        skill_probability_chart_data = pd.DataFrame.from_dict(
            dict(sorted(predicted_skill[1].items(), key=lambda item: item[1], reverse=True)), orient='index', columns=['probability']
        ).sort_values(by='probability', ascending=True)

        topic_chart, skill_chart = st.columns([40,60])

        # Build and display charts
        topic_chart_altair = build_colored_bar_chart(topic_probability_chart_data, "Topic Probabilities")
        skill_chart_altair = build_colored_bar_chart(skill_probability_chart_data, "Skill Probabilities")
        topic_chart.altair_chart(topic_chart_altair, use_container_width=True)
        skill_chart.altair_chart(skill_chart_altair, use_container_width=True)


def main():
    load_models(models, device)
    load_streamlit_ui()

    # process = psutil.Process(os.getpid())
    # while True:
    #     start_time = time.perf_counter()
    #     predicted_topic = classify_topic(problem)
    #     print(f"\t\033[92m{time.perf_counter()-start_time:.3f}s\033[0m | Predicted topic '{predicted_topic}'")
    #     st.write(f"Predicted topic: {predicted_topic}")

    #     start_time = time.perf_counter()
    #     predicted_skill = classify_skill(problem, predicted_topic)
    #     print(f"\t\033[92m{time.perf_counter()-start_time:.3f}s\033[0m | Predicted skill '{predicted_skill}'")
    #     st.write(f"Predicted skill: {predicted_skill}")

    #     print(f"\tMemory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB\n")
    #     st.write(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

if __name__ == "__main__":
    device = "cpu"
    models = {
        "topic_classifier": {
            "model": BertForSequenceClassification.from_pretrained("models/topic_classifier"),
            "tokenizer": BertTokenizer.from_pretrained("models/topic_classifier"),
            "labels": {
                0: "algebra",
                1: "geometry",
                2: "number_theory",
                3: "combinatorics"
            }
        },
        "algebra_classifier": {
            "model": BertForSequenceClassification.from_pretrained("models/algebra_classifier_8158_epoch12_0729_21-45-15"),
            "tokenizer": BertTokenizer.from_pretrained("models/algebra_classifier_8158_epoch12_0729_21-45-15"),
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
            "model": BertForSequenceClassification.from_pretrained("models/geometry_classifier_8435_epoch6_0729_20-40-41"),
            "tokenizer": BertTokenizer.from_pretrained("models/geometry_classifier_8435_epoch6_0729_20-40-41"),
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
            "model": BertForSequenceClassification.from_pretrained("models/number_theory_classifier_7109_epoch6_0729_20-34-55"),
            "tokenizer": BertTokenizer.from_pretrained("models/number_theory_classifier_7109_epoch6_0729_20-34-55"),
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
            "model": BertForSequenceClassification.from_pretrained("models/combinatorics_classifier_7368_epoch16_0729_22-36-57"),
            "tokenizer": BertTokenizer.from_pretrained("models/combinatorics_classifier_7368_epoch16_0729_22-36-57"),
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
    main()
