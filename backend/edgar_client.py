import requests
import time
import json
from datetime import datetime
from typing import Union, List
import logging
logging.basicConfig(format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
USER_AGENT = "Personal RAG App (carroll.jac@northeastern.edu) Python/3.9"
HEADERS = {"User-Agent": USER_AGENT}
RATE_LIMIT_DELAY = 0.11  

def get_cik_from_ticker(ticker: str) -> Union[str, None]:
    """
    Fetches the CIK for a given ticker symbol.
    """
    print(f"Fetching CIK for ticker: {ticker}")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)

        ticker_map = response.json()
        for company_info in ticker_map.values():
            if company_info["ticker"].upper() == ticker.upper():
                cik = str(company_info.get("cik_str", 0)).zfill(10)
                print(f"Found CIK for {ticker}: {cik}")
                return cik
        print(f"CIK not found for ticker: {ticker}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ticker-CIK map: {e}")
        return None

def get_company_submissions(cik: str) -> Union[dict, None]:
    """
    Fetches all submission metadata for a given CIK.
    """
    print(f"Fetching submissions for CIK: {cik}")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
        print(f"Successfully fetched submissions for CIK: {cik}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching submissions for CIK {cik}: {e}")
        return None

def get_filing_document(accession_number: str, primary_document: str, cik: str) -> Union[str, None]:
    """
    Fetches the content of a specific filing document.
    """
    print(f"Fetching document: {primary_document} for accession: {accession_number}")
    accession_number_clean = accession_number.replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number_clean}/{primary_document}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
        print(f"Successfully fetched document: {primary_document}")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching document {primary_document} for accession {accession_number}: {e}")
        return None

def get_filings(ticker: str, start_date: str, end_date: str) -> List[str]:
    """
    Fetches all filings for a given ticker and date range.
    """
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return []

    submissions = get_company_submissions(cik)
    if not submissions:
        return []

    filings = []
    for i, accession_number in enumerate(submissions["filings"]["recent"]["accessionNumber"]):
        filing_date = submissions["filings"]["recent"]["filingDate"][i]
        form_type = submissions["filings"]["recent"]["form"][i]
        if start_date <= filing_date <= end_date and form_type in ["10-K", "10-Q"]:
            primary_document = submissions["filings"]["recent"]["primaryDocument"][i]
            logger.info(f"Processing filing {accession_number} dated {filing_date} type {form_type}")
            document_content = get_filing_document(accession_number, primary_document, cik)
            if document_content:
                filings.append(document_content)
    return filings
