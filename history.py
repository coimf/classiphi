import streamlit as st
import json
import uuid
import os
from datetime import datetime
from typing import Optional

@st.cache_data(show_spinner=False)
def add_history(
    problem: str,
    predicted_topic: str,
    predicted_skill: str,
    topic_classifier_model_name: str,
    skill_classifier_model_name: str,
    memory_usage_mb: float,
) -> str:
    load_persistent_history = os.path.exists('history.json')
    if load_persistent_history:
        with open('history.json', 'r') as f:
            st.session_state.history = json.load(f)
    elif "history" not in st.session_state:
        st.session_state.history = {}
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
    if load_persistent_history:
        with open('history.json', 'w') as f:
            json.dump(st.session_state.history, f)
    return entry_id

def save_feedback(id: str, feedback: int) -> None:
    load_persistent_history = os.path.exists('history.json')
    if load_persistent_history:
        with open('history.json', 'r') as f:
            st.session_state.history = json.load(f)
    elif "history" not in st.session_state:
        st.session_state.history = {}
    if id in st.session_state.history and feedback in [0,1]:
        st.session_state.history[id]['feedback'] = feedback
        if load_persistent_history:
            with open('history.json', 'w') as f:
                json.dump(st.session_state.history, f)

def increase_max_results_per_page(increment: Optional[int] = 10) -> None:
    if "max_results_per_page" in st.session_state:
        st.session_state.max_results_per_page += (increment if increment and increment > 0 else 10)

def main():
    global load_persistent_history
    load_persistent_history = os.path.exists('history.json')
    st.set_page_config(page_title="ClassiPhi | History", page_icon="👾")
    st.title("History :material/history:")
    with st.spinner("Loading history..."):
        if load_persistent_history:
            with open('history.json', 'r') as f:
                st.session_state.history = json.load(f)
        else:
            st.session_state.history = {}
        if "max_results_per_page" not in st.session_state:
            st.session_state.max_results_per_page = 10
    if len(st.session_state.history) == 0:
        st.write("No history yet. Classify a problem to get started!")
        st.page_link("inference.py", label="To Classification", icon=":material/arrow_forward_ios:")
    else:
        st.write("Problems you classify will appear here.")
        st.divider()
        columns = st.columns([65, 35])
        columns[0].write("## Problem")
        columns[1].write("## Result")
        for entry in reversed(list(st.session_state.history.values())[-st.session_state.max_results_per_page:]):
            columns = st.columns([65, 35])
            columns[0].markdown(entry['problem'])
            columns[1].code(entry['predicted_topic'], language=None, wrap_lines=True, height="stretch")
            columns[1].code(entry['predicted_skill'], language=None, wrap_lines=True, height="stretch")
            st.divider()
        if(len(st.session_state.history.values()) > st.session_state.max_results_per_page):
            st.button("Load More", key="load_more", on_click=increase_max_results_per_page)

if __name__ == "__main__":
    main()
