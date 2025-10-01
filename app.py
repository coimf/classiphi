import streamlit as st

if __name__ == "__main__":
    pages = [
        st.Page(
            "inference.py",
            title="Classification",
            icon=":material/arrow_split:"
        ),
        st.Page(
            "history.py",
            title="History",
            icon=":material/history:"
        ),
    ]
    page = st.navigation(pages)
    page.run()
