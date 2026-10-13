"""Secure file text extractor for PDF and TXT news articles.

Supports:
  - Text files (.txt): UTF-8, UTF-8 with BOM (utf-8-sig), with automatic encoding fallback.
  - PDF files (.pdf): multipage extraction via pypdf with scanned PDF detection and page caps.

Security & Validation:
  - Enforces 10 MB maximum file size limit.
  - Caps PDF page processing to prevent denial of service.
  - Detects password-protected, encrypted, or corrupted files gracefully.
  - Rejects empty documents or files with insufficient readable text (< 20 characters).
  - Sanitizes errors so internal paths are not exposed to users.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import pypdf

# Maximum file size allowed (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Maximum PDF pages to process to avoid excessive processing time
MAX_PDF_PAGES = 100

# Minimum readable text length to be considered a valid article
MIN_TEXT_LENGTH = 20

# Supported extensions
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def extract_from_txt(file_bytes: bytes, filename: str = "document.txt") -> tuple[bool, str, str]:
    """Extract and decode text from a TXT file.

    Handles UTF-8, UTF-8 with BOM, and falls back to latin-1 if decoding fails.

    Returns:
        (success: bool, extracted_text_or_empty: str, status_or_error_message: str)
    """
    if not file_bytes:
        return False, "", "The uploaded text file is empty (0 bytes)."

    # Attempt decodings in order of likelihood
    decoded_text = None
    encodings_to_try = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

    for enc in encodings_to_try:
        try:
            decoded_text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        # Final fallback with character replacement
        decoded_text = file_bytes.decode("utf-8", errors="replace")

    clean_text = decoded_text.strip()

    if len(clean_text) < MIN_TEXT_LENGTH:
        return (
            False,
            "",
            f"The text file '{filename}' contains insufficient readable text ({len(clean_text)} characters). "
            f"Please provide an article with at least {MIN_TEXT_LENGTH} characters.",
        )

    return (
        True,
        clean_text,
        f"Successfully extracted {len(clean_text):,} characters from '{filename}'.",
    )


def extract_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> tuple[bool, str, str]:
    """Extract selectable text from a PDF file using pypdf.

    Handles multipage documents, page bounds, encryption, and detects scanned/image-only PDFs.

    Returns:
        (success: bool, extracted_text_or_empty: str, status_or_error_message: str)
    """
    if not file_bytes:
        return False, "", "The uploaded PDF file is empty (0 bytes)."

    # Basic header check for PDF magic bytes
    if not file_bytes.startswith(b"%PDF"):
        return False, "", f"The file '{filename}' does not appear to be a valid PDF document."

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        return False, "", f"Unable to parse '{filename}'. The PDF file may be corrupted."

    # Handle encryption / password protection
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
            if decrypted == pypdf.PasswordType.NOT_DECRYPTED:
                return (
                    False,
                    "",
                    f"The PDF '{filename}' is password-protected or encrypted. Please provide an unencrypted document.",
                )
        except Exception:
            return (
                False,
                "",
                f"The PDF '{filename}' is password-protected or encrypted and cannot be opened.",
            )

    total_pages = len(reader.pages)
    if total_pages == 0:
        return False, "", f"The PDF '{filename}' contains no pages."

    pages_to_process = min(total_pages, MAX_PDF_PAGES)
    extracted_page_texts: list[str] = []

    try:
        for page_idx in range(pages_to_process):
            page = reader.pages[page_idx]
            page_text = page.extract_text() or ""
            page_text_clean = page_text.strip()
            if page_text_clean:
                extracted_page_texts.append(page_text_clean)
    except Exception:
        return (
            False,
            "",
            f"An error occurred while reading pages from '{filename}'. The PDF structure may be damaged.",
        )

    full_text = "\n\n".join(extracted_page_texts).strip()

    # Detect scanned or image-only PDFs
    if len(full_text) < MIN_TEXT_LENGTH:
        return (
            False,
            "",
            f"No readable text could be extracted from '{filename}'. "
            "This document appears to be a scanned image or contains non-selectable text. "
            "Please provide a PDF with digital, selectable text or type/paste the text manually.",
        )

    note = ""
    if total_pages > MAX_PDF_PAGES:
        note = f" (Processed first {MAX_PDF_PAGES} of {total_pages} pages)"

    return (
        True,
        full_text,
        f"Successfully extracted {len(full_text):,} characters from {pages_to_process} page(s) of '{filename}'.{note}",
    )


def extract_text_from_file(
    file_source: Union[bytes, BinaryIO, Path, str, object],
    filename: str,
) -> tuple[bool, str, str]:
    """Universal dispatcher to extract article text from uploaded PDF or TXT files.

    Args:
        file_source: Bytes, file-like object, or file path.
        filename: Name of the file with extension.

    Returns:
        (success: bool, extracted_text: str, message: str)
    """
    if not filename:
        return False, "", "Filename was not provided."

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return (
            False,
            "",
            f"Unsupported file format '{ext}'. Supported formats are: {supported_str}.",
        )

    # Read bytes from file_source
    try:
        if isinstance(file_source, bytes):
            file_bytes = file_source
        elif hasattr(file_source, "read"):
            # File-like object (e.g. Streamlit UploadedFile or BytesIO)
            # Reset seek position if available
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            file_bytes = file_source.read()
        elif isinstance(file_source, (str, Path)):
            p = Path(file_source)
            if not p.exists():
                return False, "", "The specified file does not exist."
            file_bytes = p.read_bytes()
        else:
            return False, "", "Invalid file object supplied."
    except Exception:
        return False, "", "Failed to read the uploaded file contents."

    # Enforce size limit
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        file_mb = len(file_bytes) / (1024 * 1024)
        return (
            False,
            "",
            f"File size ({file_mb:.1f} MB) exceeds the maximum allowed limit of {max_mb} MB.",
        )

    if ext == ".txt":
        return extract_from_txt(file_bytes, filename=filename)
    elif ext == ".pdf":
        return extract_from_pdf(file_bytes, filename=filename)

    return False, "", f"Unsupported file type: {ext}"
