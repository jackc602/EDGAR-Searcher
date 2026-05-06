"""
Document chunker for SEC filings.

Parses HTML filings and splits them into chunks with metadata.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from bs4 import BeautifulSoup

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

    SEC filings reference each item at least twice — once in the table of
    contents and once as the actual section header. We pick the second
    occurrence so the section captures real content rather than the TOC.
    """
    positions_by_item: dict[str, List[int]] = {}
    for match in ITEM_PATTERN.finditer(text):
        item_number = match.group(2).upper()
        positions_by_item.setdefault(item_number, []).append(match.start())

    sections = []
    for item_number, positions in positions_by_item.items():
        start = positions[1] if len(positions) > 1 else positions[0]
        item_name = ITEM_NAMES.get(item_number, "Unknown Section")
        sections.append((start, item_number, item_name))

    sections.sort(key=lambda x: x[0])
    return sections


# Sentence boundary: period/!/? followed by whitespace, then a capital letter
# or digit. Good enough for SEC English prose without pulling in nltk.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


def _split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    return sentences


def split_text_into_chunks(
    text: str,
    target_chars: int = 1100,
    overlap_sentences: int = 1,
) -> List[Tuple[str, int]]:
    """
    Pack sentences into chunks up to a target character budget.

    target_chars ≈ 1100 ≈ ~275 tokens, comfortably under the 512-token cap of
    mxbai-embed-large. overlap_sentences carries a tail of full sentences from
    one chunk into the start of the next so semantic context is preserved at
    boundaries.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Tuple[str, int]] = []
    chunk_index = 0
    buffer: List[str] = []
    buffer_len = 0

    for sentence in sentences:
        # If a single sentence overflows the budget, hard-wrap it on whitespace
        # to avoid producing one giant chunk that the embedder will truncate.
        if len(sentence) > target_chars:
            if buffer:
                chunks.append((' '.join(buffer), chunk_index))
                chunk_index += 1
                buffer = []
                buffer_len = 0
            words = sentence.split()
            piece: List[str] = []
            piece_len = 0
            for word in words:
                added = len(word) + (1 if piece else 0)
                if piece_len + added > target_chars:
                    chunks.append((' '.join(piece), chunk_index))
                    chunk_index += 1
                    piece = [word]
                    piece_len = len(word)
                else:
                    piece.append(word)
                    piece_len += added
            if piece:
                buffer = piece
                buffer_len = piece_len
            continue

        added = len(sentence) + (1 if buffer else 0)
        if buffer_len + added > target_chars and buffer:
            chunks.append((' '.join(buffer), chunk_index))
            chunk_index += 1
            tail = buffer[-overlap_sentences:] if overlap_sentences > 0 else []
            buffer = list(tail)
            buffer_len = sum(len(s) for s in buffer) + max(0, len(buffer) - 1)
            added = len(sentence) + (1 if buffer else 0)

        buffer.append(sentence)
        buffer_len += added

    if buffer:
        chunks.append((' '.join(buffer), chunk_index))

    return chunks


def chunk_filing(
    html_content: str,
    ticker: str,
    cik: str,
    accession_number: str,
    filing_date: str,
    filing_type: str,
    target_chars: int = 1100,
    overlap_sentences: int = 1,
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
        target_chars: Approximate character budget per chunk (~275 tokens at 1100).
        overlap_sentences: Number of trailing sentences carried into the next chunk.

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
        text_chunks = split_text_into_chunks(text, target_chars, overlap_sentences)
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
            text_chunks = split_text_into_chunks(section_text, target_chars, overlap_sentences)
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
