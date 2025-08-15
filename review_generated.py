import streamlit as st
import json
from streamlit_extras.stylable_container import stylable_container

def load_streamlit_ui() -> None:
    st.title("Dataset Review")
    if 'current_test_index' not in st.session_state:
        st.session_state.current_test_index = 0
    if 'current_problem_index' not in st.session_state:
        st.session_state.current_problem_index = 0
    if 'show_decline_form' not in st.session_state:
        st.session_state.show_decline_form = False
    data = load_data()
    if not st.session_state.current_problem_keys or st.session_state.current_problem_keys != list(data.keys()):
        st.session_state.current_problem_index = 0
        for p in data:
            if p['reviewed']:
                st.session_state.current_problem_index += 1
                continue
            break
    st.markdown(f"{st.session_state.current_problem_index}/{1000} | {st.session_state.current_problem_index/10:.2f}% complete")
    st.divider()
    for n, p in enumerate(data):
        if not p['reviewed']:
            with st.container():
                with st.expander("Problem Details", expanded=True):
                    st.markdown("**Problem:**")
                    st.markdown(p['problem'])
                    st.markdown(f"**Solution:** {p['solution']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Topic:** {p['topic']}")
                    with col2:
                        st.markdown(f"**Skill:** {p['skill']}")
                    st.markdown(f"**Author:** {p['source'] if p['source'] else 'Human'}")
                st.markdown("**Review Decision:**")
                with stylable_container(
                    key=f"buttons_{n}",
                    css_styles="""
                    div[data-testid="column"] {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                    }
                    button[kind="secondary"] {
                        background-color: #e8a9ad !important;
                        color: white !important;
                        border: none !important;
                        border-radius: 12px !important;
                        transition: all 0.3s ease !important;
                    }
                    button[kind="secondary"]:hover {
                        background-color: #d4817a !important;
                        transform: translateY(-2px) !important;
                        box-shadow: 0 6px 20px rgba(232, 153, 141, 0.4) !important;
                    }
                    button[kind="primary"] {
                        background-color: #7eb3a3 !important;
                        border: none !important;
                        border-radius: 12px !important;
                        transition: all 0.3s ease !important;
                    }
                    button[kind="primary"]:hover {
                        background-color: #6bbe91 !important;
                        transform: translateY(-2px) !important;
                        box-shadow: 0 6px 20px rgba(126, 179, 163, 0.4) !important;
                    }
                    """
                ):
                    col1, col2, col3 = st.columns([2, 3, 2])
                    with col1:
                        decline = st.button(
                            "Decline",
                            key=f"decline_{n}",
                            type="secondary",
                            use_container_width=True,
                        )
                    with col3:
                        accept = st.button(
                            "Accept",
                            key=f"accept_{n}",
                            type="primary",
                            use_container_width=True,
                        )
                st.divider()
                labels = {
                    "algebra": [
                        "vieta's formulas",
                        "factoring tricks",
                        "floor/ceiling functions",
                        "distance/work rate and time",
                        "inequalities",
                        "fractions ratios and percents",
                        "polynomials",
                        "arithmetic/geometric sequences",
                        "graphing equations",
                        "systems of equations",
                        "logarithm properties",
                        "complex numbers",
                        "trigonometric identities",
                        "telescoping series"
                    ],
                    "geometry": [
                        "pythagorean theorem",
                        "similar/congruent triangles",
                        "inscribed/circumscribed circles",
                        "power of a point",
                        "coordinate bashing",
                        "3D geometry",
                        "area/volume formulas",
                        "perimeter"
                    ],
                    "number theory": [
                        "modular arithmetic",
                        "divisibility rules",
                        "prime factorization",
                        "diophantine equations",
                        "base conversion",
                        "GCD and LCM",
                        "Euler's Theorem",
                        "exponent cycles"
                    ],
                    "counting": [
                        "constructive counting",
                        "casework",
                        "complementary counting",
                        "principle of inclusion-exclusion",
                        "stars and bars",
                        "pigeonhole principle"
                    ],
                    "probability": [
                        "expected value",
                        "casework",
                        "geometric probability",
                        "conditional probability",
                        "independent events",
                        "constructive probability",
                        "states"
                    ],
                }
                if accept:
                    p['reviewed'] = True
                    save_data(data)
                    st.session_state.current_problem_index += 1
                    st.session_state.show_decline_form = False
                    st.rerun()
                elif decline:
                    st.session_state.show_decline_form = True
                    st.rerun()

                if st.session_state.show_decline_form:
                    options = [skill for skill in labels[v['label']['topic']] if skill != v['label']['skill']]
                    options.append("no suitable skill")
                    with st.form(f"override_form_{st.session_state.current_problem_index}"):
                        override = st.selectbox(
                            "Choose an override label",
                            placeholder="Select label...",
                            index=None,
                            options=options
                        )
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            cancel = st.form_submit_button("Cancel", type="secondary", use_container_width=True)
                        with col3:
                            confirmation = st.form_submit_button("Confirm", type="primary", use_container_width=True)

                        if cancel:
                            st.session_state.show_decline_form = False
                            st.rerun()
                        elif confirmation and override is not None:
                            v['label']['skill'] = override
                            v['label']['reviewed'] = True
                            save_data(test, data)
                            st.session_state.current_problem_index += 1
                            st.session_state.show_decline_form = False
                            st.rerun()
                        elif confirmation and override is None:
                            st.error("Please select an override label before confirming.")

def load_data() -> dict:
    with open('training_data_1k.json', 'r') as f:
        return json.load(f)

def save_data(data: dict) -> None:
    with open('training_data_1k', 'w') as f:
        json.dump(data, f, indent=2)

def main():
    st.set_page_config(page_title="ClassiPhi | Review Dataset", page_icon="👾")
    load_streamlit_ui()

if __name__ == "__main__":
    main()
