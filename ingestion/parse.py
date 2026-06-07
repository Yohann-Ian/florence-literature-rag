"""
ingestion/parse.py

Florence — Literature RAG Agent
Parses EPUB and PDF files from the corpus into clean text documents.
Supports both formats to handle the mixed corpus (EPUB + one PDF).
"""

import os
import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import Optional


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class ParsedDocument:
    """
    A single parsed book from the corpus.
    Contains the full clean text and metadata Florence needs for citation.
    """
    title: str
    author: str
    filename: str
    text: str
    format: str  # 'epub' or 'pdf'
    char_count: int
    word_count: int


# ── EPUB parser ───────────────────────────────────────────────────────────────

def parse_epub(filepath: str) -> str:
    """
    Extract clean text from an EPUB file.
    Uses ebooklib to read chapters and BeautifulSoup to strip HTML tags.
    """
    book = epub.read_epub(filepath, options={"ignore_ncx": True})
    chapters = []

    for item in book.get_items():
        # Only process document items — skips images, CSS, fonts
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Remove tags that aren't prose — notes, scripts, styles
            for tag in soup(["script", "style", "head", "title", "meta"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            text = clean_text(text)

            if len(text.strip()) > 100:  # skip near-empty chapters
                chapters.append(text)

    return "\n\n".join(chapters)


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf(filepath: str) -> str:
    """
    Extract clean text from a PDF file using PyMuPDF.
    Handles the Master and Margarita PDF in the corpus.
    """
    doc = fitz.open(filepath)
    pages = []

    for page in doc:
        text = page.get_text("text")
        text = clean_text(text)
        if len(text.strip()) > 50:  # skip near-empty pages
            pages.append(text)

    doc.close()
    return "\n\n".join(pages)


# ── Text cleaner ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise extracted text.
    Removes artifacts common in both EPUB and PDF extraction.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive whitespace and blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove page number artifacts (common in PDF extraction)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


# ── Metadata registry ─────────────────────────────────────────────────────────

# Maps filename (without extension) to author metadata
# Update this if filenames change
CORPUS_METADATA = {
    "Anna Karenina":                    "Leo Tolstoy",
    "Beloved":                          "Toni Morrison",
    "Brave New World":                  "Aldous Huxley",
    "Crime and Punishment":             "Fyodor Dostoyevsky",
    "East of Eden":                     "John Steinbeck",
    "Farewell to Arms":                 "Ernest Hemingway",
    "Lectures on Russian literature":   "Vladimir Nabokov",
    "Master and Margarita":             "Mikhail Bulgakov",
    "Nineteen Eighty-Four":             "George Orwell",
    "Paradise Lost":                    "John Milton",
    "The Aeneid":                       "Virgil",
    "Aspects of the Novel":             "E.M. Forster",
}


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_document(filepath: str) -> Optional[ParsedDocument]:
    """
    Parse a single document from the corpus.
    Detects format from file extension and routes to the correct parser.
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1].lower()

    # Look up author from metadata registry
    author = CORPUS_METADATA.get(name_without_ext, "Unknown")

    print(f"  Parsing: {filename} ({extension})")

    try:
        if extension == ".epub":
            text = parse_epub(filepath)
            fmt = "epub"
        elif extension == ".pdf":
            text = parse_pdf(filepath)
            fmt = "pdf"
        else:
            print(f"  Skipping unsupported format: {extension}")
            return None

        word_count = len(text.split())
        char_count = len(text)

        print(f"  Done: {word_count:,} words, {char_count:,} characters")

        return ParsedDocument(
            title=name_without_ext,
            author=author,
            filename=filename,
            text=text,
            format=fmt,
            char_count=char_count,
            word_count=word_count,
        )

    except Exception as e:
        print(f"  ERROR parsing {filename}: {e}")
        return None


def parse_corpus(corpus_dir: str) -> list[ParsedDocument]:
    """
    Parse all documents in the corpus directory.
    Returns a list of ParsedDocument objects, one per book.
    """
    print(f"\nParsing corpus from: {corpus_dir}")
    print("=" * 60)

    documents = []
    supported = {".epub", ".pdf"}

    files = sorted([
        f for f in os.listdir(corpus_dir)
        if os.path.splitext(f)[1].lower() in supported
    ])

    if not files:
        print("No supported files found in corpus directory.")
        return documents

    for filename in files:
        filepath = os.path.join(corpus_dir, filename)
        doc = parse_document(filepath)
        if doc:
            documents.append(doc)

    print("=" * 60)
    print(f"Corpus parsed: {len(documents)} documents")
    total_words = sum(d.word_count for d in documents)
    print(f"Total words across corpus: {total_words:,}")

    return documents


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to test the parser against the corpus.
    Usage: python ingestion/parse.py
    """
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Books"
    )

    if not os.path.exists(corpus_path):
        print(f"Corpus directory not found: {corpus_path}")
        print("Make sure your Books/ folder is in the project root.")
    else:
        docs = parse_corpus(corpus_path)
        print("\nSample from first document:")
        print("-" * 40)
        if docs:
            print(docs[0].text[:500])