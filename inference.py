import streamlit as st
st.set_page_config(page_title="Math Problem Classifier", page_icon="👾")
st.title("Math Problem Topic and Skill Classifier")
with st.spinner("Loading models...", show_time=True):
    import time
    import psutil
    import os
    import re
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer
    from streamlit_extras.stylable_container import stylable_container

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

def classify_topic(problem: str) -> str:
    """
    Classifies topic of the given problem using topic classifier model.

    Args:
        problem (str): The problem to classify.

    Returns:
        str: The predicted topic of the problem.
    """
    global models
    input_ids = models['topic_classifier']['tokenizer'](problem, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = models['topic_classifier']['model'](**input_ids)
    logits = outputs.logits
    predicted_topic_id = torch.argmax(logits, dim=-1).item()
    predicted_topic = models['topic_classifier']['labels'][predicted_topic_id].capitalize()
    return predicted_topic

def classify_skill(problem: str, topic: str) -> str:
    """
    Classifies skill of the given problem using skill classifier model.

    Args:
        problem (str): The problem to classify.
        topic (str): The topic of the problem.

    Returns:
        str: The predicted skill of the problem.
    """
    global models
    skill_classifier_name = f"{topic}_classifier".lower()
    input_ids = models[skill_classifier_name]['tokenizer'](problem, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = models[skill_classifier_name]['model'](**input_ids)
    logits = outputs.logits
    predicted_skill_id = torch.argmax(logits, dim=-1).item()
    predicted_skill = models[skill_classifier_name]['labels'][predicted_skill_id].capitalize()
    return predicted_skill

def load_streamlit_ui():
    problem = st.text_area("Enter a problem for classification:", placeholder="Paste a problem here...", height="content")
    with stylable_container(key="classify", css_styles=r"""
        button {
            float: right;
            margin-bottom: 20px;
        }
    """):
        start_classification = st.button("Classify", type="primary")

    if start_classification:
        if problem:
            with st.spinner("Classifying...", show_time=True):
                predicted_topic = classify_topic(problem)
                predicted_skill = classify_skill(problem, predicted_topic)
            if '$' in problem:
                st.markdown(problem)
            st.write(f"Predicted topic: {predicted_topic}")
            st.write(f"Predicted skill: {predicted_skill}")
        else:
            st.info("Please enter a problem first.")

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
