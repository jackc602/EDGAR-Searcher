"""
Document chunker for SEC filings.

Parses HTML filings and splits them into chunks with metadata.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from bs4 import BeautifulSoup

logging.basicConfig(
    format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Regex pattern to identify Item boundaries in SEC filings
ITEM_PATTERN = re.compile(
    r'(Item|ITEM)\s*(1A|1B|1C|2|3|4|5|6|7A|7|8|9A|9B|9|10|11|12|13|14|15|1)\b',
    re.IGNORECASE
)

# Mapping of item numbers to their standard names
ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}


@dataclass
class DocumentChunk:
    """Represents a chunk of a SEC filing document with metadata."""
    text: str
    item_number: Optional[str]
    item_name: Optional[str]
    chunk_index: int
    filing_type: str
    filing_date: str
    ticker: str
    cik: str
    accession_number: str

    def to_metadata_dict(self) -> dict:
        """Convert metadata to dict for Chroma storage."""
        return {
            "item_number": self.item_number or "",
            "item_name": self.item_name or "",
            "chunk_index": self.chunk_index,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "ticker": self.ticker,
            "cik": self.cik,
            "accession_number": self.accession_number,
        }

    def generate_id(self) -> str:
        """Generate a unique ID for this chunk."""
        item = self.item_number or "unknown"
        return f"{self.accession_number}_{item}_{self.chunk_index}"


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content using BeautifulSoup.

    Args:
        html_content: Raw HTML string from SEC filing.

    Returns:
        Clean text with excessive whitespace removed.
    """
    soup = BeautifulSoup(html_content, 'lxml')

    # Remove script and style elements
    for element in soup(['script', 'style', 'head', 'meta', 'link']):
        element.decompose()

    # Get text and clean whitespace
    text = soup.get_text(separator=' ')
    # Collapse multiple whitespace characters into single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def identify_item_sections(text: str) -> List[Tuple[int, str, str]]:
    """
    Identify Item section boundaries in the text.

    Args:
        text: Clean text from SEC filing.

    Returns:
        List of tuples (start_position, item_number, item_name).
    """
    sections = []
    seen_items = set()

    for match in ITEM_PATTERN.finditer(text):
        item_number = match.group(2).upper()
        # Normalize item number (e.g., "1a" -> "1A")
        item_number = item_number.upper()

        # Skip if we've already seen this item (avoid duplicates from TOC)
        if item_number in seen_items:
            continue
        seen_items.add(item_number)

        item_name = ITEM_NAMES.get(item_number, "Unknown Section")
        sections.append((match.start(), item_number, item_name))

    # Sort by position
    sections.sort(key=lambda x: x[0])
    return sections


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[Tuple[str, int]]:
    """
    Split text into word-based chunks with overlap.

    Args:
        text: Text to split.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of tuples (chunk_text, chunk_index).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    chunk_index = 0
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append((chunk_text, chunk_index))
        chunk_index += 1

        # Move start forward, accounting for overlap
        start = end - overlap if end < len(words) else len(words)

        # Prevent infinite loop if overlap >= chunk_size
        if start <= chunks[-1][1] * (chunk_size - overlap) and end < len(words):
            start = end

    return chunks


def chunk_filing(
    html_content: str,
    ticker: str,
    cik: str,
    accession_number: str,
    filing_date: str,
    filing_type: str,
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[DocumentChunk]:
    """
    Parse and chunk a SEC filing into DocumentChunks with metadata.

    Args:
        html_content: Raw HTML content of the filing.
        ticker: Company ticker symbol.
        cik: CIK number.
        accession_number: SEC accession number.
        filing_date: Date of the filing (YYYY-MM-DD).
        filing_type: Type of filing (10-K, 10-Q).
        chunk_size: Target words per chunk.
        overlap: Overlapping words between chunks.

    Returns:
        List of DocumentChunk objects.
    """
    logger.info(f"Chunking filing {accession_number} for {ticker}")

    # Extract clean text
    text = extract_text_from_html(html_content)
    if not text:
        logger.warning(f"No text extracted from filing {accession_number}")
        return []

    # Identify item sections
    sections = identify_item_sections(text)
    logger.info(f"Found {len(sections)} item sections in filing")

    chunks = []
    global_chunk_index = 0

    if not sections:
        # No sections found, chunk the entire document
        text_chunks = split_text_into_chunks(text, chunk_size, overlap)
        for chunk_text, _ in text_chunks:
            chunk = DocumentChunk(
                text=chunk_text,
                item_number=None,
                item_name=None,
                chunk_index=global_chunk_index,
                filing_type=filing_type,
                filing_date=filing_date,
                ticker=ticker,
                cik=cik,
                accession_number=accession_number,
            )
            chunks.append(chunk)
            global_chunk_index += 1
    else:
        # Process each section
        for i, (start_pos, item_number, item_name) in enumerate(sections):
            # Determine end position (start of next section or end of text)
            if i + 1 < len(sections):
                end_pos = sections[i + 1][0]
            else:
                end_pos = len(text)

            section_text = text[start_pos:end_pos].strip()
            if not section_text:
                continue

            # Chunk this section
            text_chunks = split_text_into_chunks(section_text, chunk_size, overlap)
            for chunk_text, _ in text_chunks:
                chunk = DocumentChunk(
                    text=chunk_text,
                    item_number=item_number,
                    item_name=item_name,
                    chunk_index=global_chunk_index,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    ticker=ticker,
                    cik=cik,
                    accession_number=accession_number,
                )
                chunks.append(chunk)
                global_chunk_index += 1

    logger.info(f"Created {len(chunks)} chunks for filing {accession_number}")
    return chunks
