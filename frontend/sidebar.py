import streamlit as st

def load_sidebar():
    st.sidebar.title("Navigation")
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/1_Filings_Viewer.py", label="Filings Viewer")
