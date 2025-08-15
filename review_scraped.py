import streamlit as st
import json
from streamlit_extras.stylable_container import stylable_container
import streamlit.components.v1 as components

def add_keyboard_shortcuts(problem_key: str):
    keyboard_js = f"""
    <script>
    document.addEventListener('keydown', function(event) {{
        if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {{
            if (event.key === 'a' || event.key === 'A') {{
                // Find and click the decline button
                const declineButton = document.querySelector('[data-testid="baseButton-secondary"][key="decline_{problem_key}"]');
                if (declineButton) {{
                    declineButton.click();
                    event.preventDefault();
                }}
            }} else if (event.key === 'd' || event.key === 'D') {{
                // Find and click the accept button
                const acceptButton = document.querySelector('[data-testid="baseButton-primary"][key="accept_{problem_key}"]');
                if (acceptButton) {{
                    acceptButton.click();
                    event.preventDefault();
                }}
            }}
        }}
    }});
    document.addEventListener('keydown', function(event) {{
        if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {{
            if (event.key === 'a' || event.key === 'A') {{
                const buttons = document.querySelectorAll('button');
                for (let button of buttons) {{
                    if (button.textContent.trim() === 'Decline') {{
                        button.click();
                        event.preventDefault();
                        break;
                    }}
                }}
            }} else if (event.key === 'd' || event.key === 'D') {{
                const buttons = document.querySelectorAll('button');
                for (let button of buttons) {{
                    if (button.textContent.trim() === 'Accept') {{
                        button.click();
                        event.preventDefault();
                        break;
                    }}
                }}
            }}
        }}
    }});
    </script>
    """
    components.html(keyboard_js, height=0)

def load_streamlit_ui() -> None:
    st.title("Dataset Review")
    tests = ['amc8', 'amc10', 'amc12', 'aime']
    if 'current_test_index' not in st.session_state:
        st.session_state.current_test_index = 0
    if 'current_problem_keys' not in st.session_state:
        st.session_state.current_problem_keys = []
    if 'current_problem_index' not in st.session_state:
        st.session_state.current_problem_index = 0
    if 'show_decline_form' not in st.session_state:
        st.session_state.show_decline_form = False
    if st.session_state.current_test_index >= len(tests):
        st.success("Dataset Review Complete!")
        return
    test = tests[st.session_state.current_test_index]
    data = load_data(test)

    if not data:
        st.info(f"No problems found for {test.upper()}")
        st.session_state.current_test_index += 1
        st.session_state.current_problem_keys = []
        st.session_state.current_problem_index = 0
        st.rerun()

    if not st.session_state.current_problem_keys or st.session_state.current_problem_keys != list(data.keys()):
        st.session_state.current_problem_keys = list(data.keys())
        st.session_state.current_problem_index = 0
        for n, k in enumerate(st.session_state.current_problem_keys):
            v = data[k]
            if not v['label']['reviewed']:
                st.session_state.current_problem_index = n
                break

    if st.session_state.current_problem_index >= len(st.session_state.current_problem_keys):
        st.session_state.current_test_index += 1
        st.session_state.current_problem_keys = []
        st.session_state.current_problem_index = 0
        st.rerun()

    k = st.session_state.current_problem_keys[st.session_state.current_problem_index]
    v = data[k]
    total = len(st.session_state.current_problem_keys)
    # add_keyboard_shortcuts(k)

    st.markdown(f"{st.session_state.current_problem_index}/{total} | {100*st.session_state.current_problem_index/total:.2f}% complete")
    st.divider()
    with st.container():
        st.markdown(f"## {v['title']}")
        st.page_link(f"https://artofproblemsolving.com/wiki/index.php/{v['title']}", label="View on Wiki", icon=":material/open_in_new:")
        with st.expander("Problem Details", expanded=True):
            st.markdown("**Problem:**")
            st.markdown(v['problem'])
            if v.get('answer_choices'):
                st.markdown("**Answer Choices**")
                st.markdown(v['answer_choices'])
            if v.get('answer'):
                st.markdown("**Answer**")
                st.markdown(f"${v['answer']}$")
            st.markdown(f"**Solution:** {v['solutions'][0]}")
            if v.get('label'):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Topic:** {v['label'].get('topic', 'N/A')}")
                with col2:
                    st.markdown(f"**Skill:** {v['label'].get('skill', 'N/A')}")
            st.markdown(f"**Reason:** {v['label'].get('reason', 'N/A')}")
            st.markdown(f"**Annotator:** {v['label'].get('annotator', 'N/A')}")
        st.markdown("**Review Decision:**")
        with stylable_container(
            key=f"buttons_{k}",
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
                    key=f"decline_{k}",
                    type="secondary",
                    use_container_width=True,
                )
            with col3:
                accept = st.button(
                    "Accept",
                    key=f"accept_{k}",
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
            v['label']['reviewed'] = True
            save_data(test, data)
            st.session_state.current_problem_index += 1
            st.session_state.show_decline_form = False
            st.rerun()
        elif decline:
            st.session_state.show_decline_form = True
            st.rerun()

        if st.session_state.show_decline_form:
            st.markdown(f"### Override {v['label']['annotator']}'s decision")
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

def load_data(test: str) -> dict:
    with open(f'scraped_data/problems/labeled_{test}_problems.json', 'r') as f:
        return json.load(f)

def save_data(test: str, data: dict) -> None:
    with open(f'scraped_data/problems/labeled_{test}_problems.json', 'w') as f:
        json.dump(data, f, indent=2)

def main():
    st.set_page_config(page_title="ClassiPhi | Review Dataset", page_icon="👾")
    load_streamlit_ui()

if __name__ == "__main__":
    main()
