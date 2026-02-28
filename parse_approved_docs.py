"""
parse_approved_docs.py

Parses downloaded Approved Document PDFs and chunks them into
JSON records ready to be loaded into a vector store.

Usage:
    python3 parse_approved_docs.py
"""

import os
import re
import json
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("Please install pymupdf:  pip3 install pymupdf")


# ---------------------------------------------------------------------------
# Configuration - edit these if you want
# ---------------------------------------------------------------------------

INPUT_DIR = "approved_docs"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE = 300    # target words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks


# ---------------------------------------------------------------------------
# Part metadata
# ---------------------------------------------------------------------------

PART_METADATA = {
    "ALL":  {"title": "Merged Approved Documents", "topic": "All parts"},
    "A":    {"title": "Approved Document A",        "topic": "Structure"},
    "B1":   {"title": "Approved Document B Vol.1",  "topic": "Fire safety - Dwellings"},
    "B2":   {"title": "Approved Document B Vol.2",  "topic": "Fire safety - Other buildings"},
    "C":    {"title": "Approved Document C",         "topic": "Contamination and moisture resistance"},
    "D":    {"title": "Approved Document D",         "topic": "Toxic substances"},
    "E":    {"title": "Approved Document E",         "topic": "Sound insulation"},
    "F1":   {"title": "Approved Document F Vol.1",  "topic": "Ventilation - Dwellings"},
    "F2":   {"title": "Approved Document F Vol.2",  "topic": "Ventilation - Other buildings"},
    "G":    {"title": "Approved Document G",         "topic": "Sanitation and water efficiency"},
    "H":    {"title": "Approved Document H",         "topic": "Drainage and waste disposal"},
    "J":    {"title": "Approved Document J",         "topic": "Combustion appliances"},
    "K":    {"title": "Approved Document K",         "topic": "Protection from falling, collision and impact"},
    "L1":   {"title": "Approved Document L Vol.1",  "topic": "Energy efficiency - Dwellings"},
    "L2":   {"title": "Approved Document L Vol.2",  "topic": "Energy efficiency - Other buildings"},
    "M1":   {"title": "Approved Document M Vol.1",  "topic": "Access - Dwellings"},
    "M2":   {"title": "Approved Document M Vol.2",  "topic": "Access - Other buildings"},
    "O":    {"title": "Approved Document O",         "topic": "Overheating"},
    "P":    {"title": "Approved Document P",         "topic": "Electrical safety"},
    "Q":    {"title": "Approved Document Q",         "topic": "Security"},
    "R1":   {"title": "Approved Document R Vol.1",  "topic": "Electronic communications infrastructure"},
    "R2":   {"title": "Approved Document R Vol.2",  "topic": "Electronic communications infrastructure"},
    "S":    {"title": "Approved Document S",         "topic": "EV charging infrastructure"},
    "T":    {"title": "Approved Document T",         "topic": "Toilet accommodation"},
    "Reg7": {"title": "Regulation 7",               "topic": "Materials and workmanship"},
}


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path):
    pages = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages


def clean_text(text):
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def make_chunks(full_text, chunk_size, overlap):
    words = full_text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Process a single PDF
# ---------------------------------------------------------------------------

def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    part = Path(filename).stem.split("_")[0]
    meta = PART_METADATA.get(part, {"title": filename, "topic": "Unknown"})

    print(f"  Parsing: {meta['title']} ({filename})")

    pages = extract_pages(pdf_path)
    if not pages:
        print(f"  Warning: no text found in {filename}")
        return []

    # Join all pages into one big string
    full_text = "\n\n".join(clean_text(p["text"]) for p in pages)

    raw_chunks = make_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "id": f"{part}_{i:04d}",
            "text": chunk_text,
            "metadata": {
                "part": part,
                "title": meta["title"],
                "topic": meta["topic"],
                "source_file": filename,
                "chunk_index": i,
                "word_count": len(chunk_text.split()),
            }
        })

    print(f"  Done: {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pdf_files = sorted(Path(INPUT_DIR).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in '{INPUT_DIR}'. Run download_approved_docs.py first.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{INPUT_DIR}'\n")

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = process_pdf(str(pdf_path))
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saving to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"All done! {len(all_chunks)} chunks saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
