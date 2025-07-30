import streamlit as st
import json
import uuid
from datetime import datetime

def add_history(
    problem: str,
    predicted_topic: str,
    predicted_skill: str,
    topic_classifier_model_name: str,
    skill_classifier_model_name: str,
    memory_usage_mb: float,
) -> str:
    with open('history.json', 'r') as f:
        st.session_state.history = json.load(f)
    entry_id = str(uuid.uuid4())
    st.session_state.history[entry_id] = {
        "id": entry_id,
        "problem": problem,
        "predicted_topic": predicted_topic,
        "predicted_skill": predicted_skill,
        "topic_classifier_model_name": topic_classifier_model_name,
        "skill_classifier_model_name": skill_classifier_model_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "memory_usage_mb": memory_usage_mb,
        "feedback": None
    }
    with open('history.json', 'w') as f:
        json.dump(st.session_state.history, f)
    return entry_id

def save_feedback(id, feedback) -> None:
    with open('history.json', 'r') as f:
        st.session_state.history = json.load(f)
    if id in st.session_state.history and feedback in [0,1]:
        st.session_state.history[id]['feedback'] = feedback
    with open('history.json', 'w') as f:
        json.dump(st.session_state.history, f)

def main():
    st.set_page_config(page_title="History", page_icon="👾")
    st.title("History")
    with st.spinner("Loading history..."):
        with open('history.json', 'r') as f:
            st.session_state.history = json.load(f)
    if len(st.session_state.history) == 0:
        st.write("No history yet. Classify a problem to get started!")
        st.page_link("inference.py", label="To Classification")
    else:
        columns = st.columns([65, 35])
        columns[0].write("## Problem")
        columns[1].write("## Result")
        for entry in reversed(st.session_state.history.values()):
            columns = st.columns([65, 35])
            columns[0].markdown(entry['problem'])
            columns[1].code(entry['predicted_topic'], language=None, wrap_lines=True, height="stretch")
            columns[1].code(entry['predicted_skill'], language=None, wrap_lines=True, height="stretch")

if __name__ == "__main__":
    main()
