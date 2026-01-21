import streamlit as st
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from frontend.sidebar import load_sidebar

load_sidebar()

st.title("SEC Filings Viewer")

if "text_data" in st.session_state and st.session_state.text_data:
    # Initialize session state for pagination
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0

    st.subheader("All Extracted Filings")

    total_filings = len(st.session_state.text_data)
    filings_per_page = 1

    # Pagination buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("Previous"):
            if st.session_state.page_number > 0:
                st.session_state.page_number -= 1

    with col3:
        if st.button("Next"):
            if st.session_state.page_number < total_filings - 1:
                st.session_state.page_number += 1
    
    with col2:
        st.write(f"Page {st.session_state.page_number + 1} of {total_filings}")

    # Display the filing for the current page
    st.text_area(
        f"Filing {st.session_state.page_number + 1}",
        value=st.session_state.text_data[st.session_state.page_number],
        height=400,
        disabled=True
    )
else:
    st.warning("No filings loaded. Please load filings on the main page.")
