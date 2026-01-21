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
    logger.info(f"Fetching CIK for ticker: {ticker}")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)

        ticker_map = response.json()
        for company_info in ticker_map.values():
            if company_info["ticker"].upper() == ticker.upper():
                cik = str(company_info.get("cik_str", 0)).zfill(10)
                logger.info(f"Found CIK for {ticker}: {cik}")
                return cik
        logger.info(f"CIK not found for ticker: {ticker}")
        return None
    except requests.exceptions.RequestException as e:
        logger.info(f"Error fetching ticker-CIK map: {e}")
        return None

def get_company_submissions(cik: str) -> Union[dict, None]:
    """
    Fetches all submission metadata for a given CIK.
    """
    logger.info(f"Fetching submissions for CIK: {cik}")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
        logger.info(f"Successfully fetched submissions for CIK: {cik}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.info(f"Error fetching submissions for CIK {cik}: {e}")
        return None

def get_filing_document(accession_number: str, primary_document: str, cik: str) -> Union[str, None]:
    """
    Fetches the content of a specific filing document.
    """
    logger.info(f"Fetching document: {primary_document} for accession: {accession_number}")
    accession_number_clean = accession_number.replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number_clean}/{primary_document}"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
        logger.info(f"Successfully fetched document: {primary_document}")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.info(f"Error fetching document {primary_document} for accession {accession_number}: {e}")
        return None

def get_filings(ticker: str, start_date: str, end_date: str) -> List[str]:
    """
    Fetches all filings for a given ticker and date range.
    """
    cik = get_cik_from_ticker(ticker)
    logger.info(f"Fetching filings for ticker {ticker} from {start_date} to {end_date}")
    if not cik:
        return []
    else:
        logger.info(f"CIK for ticker {ticker} is {cik}")

    submissions = get_company_submissions(cik)
    
    if not submissions:
        logger.info(f"Submissions for CIK {cik} are empty, exiting...")
        return []
    else:
        logger.info(f"Successfully fetched submissions for CIK {cik}")

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
    logger.info(f"Total filings fetched: {len(filings)}")
    return filings
