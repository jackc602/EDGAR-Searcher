import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from frontend.sidebar import load_sidebar


logging.basicConfig(
    format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s',
    level=logging.INFO,
)


load_sidebar()

st.title("SEC Filings Viewer")

if "filings_metadata" in st.session_state and st.session_state.filings_metadata:
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0

    st.subheader("All Extracted Filings")

    filings = st.session_state.filings_metadata
    total_filings = len(filings)

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

    current_filing = filings[st.session_state.page_number]

    st.markdown(
        f"**Ticker:** {current_filing['ticker']} | "
        f"**Type:** {current_filing['filing_type']} | "
        f"**Date:** {current_filing['filing_date']}"
    )
    st.markdown(f"**Accession #:** {current_filing['accession_number']}")

    st.text_area(
        f"Filing {st.session_state.page_number + 1}",
        value=current_filing.get('content', 'Content not available'),
        height=400,
        disabled=True,
    )
else:
    st.warning("No filings loaded. Please load filings on the main page.")
