"""Corpus parsing and validation module for Asterline knowledge base.

Preserves data/corpus.txt immutability, strips confirmed pagination artifacts,
validates metadata fields, and produces structured CorpusChunk records.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Regular Expression Patterns ──────────────────────────────────────────────
HEADER_PATTERN = re.compile(r"^(\d{2})\.\s+(.+?)\s+-\s+([a-zA-Z0-9_\-]+:\d+)$")
META_FIELD_PATTERN = re.compile(r"^([a-z_]+):\s*(.*)$")
CONFIRMED_ARTIFACT_PATTERN = re.compile(r"^Batch 5 \| Complete candidate package \d+$")
SUSPECT_ARTIFACT_PATTERN = re.compile(
    r"(?:Batch \d+|candidate package|Complete candidate)", re.IGNORECASE
)

REQUIRED_METADATA_KEYS = {
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "version",
}


class CorpusError(Exception):
    """Base exception for corpus operations."""


class CorpusNotFoundError(CorpusError):
    """Raised when the corpus file cannot be found."""


class CorpusValidationError(CorpusError):
    """Raised when corpus chunks fail structural or semantic validation."""


class CorpusChunk(BaseModel):
    """A validated knowledge base document chunk."""

    chunk_id: str
    doc_id: str
    title: str
    category: str
    owner: str
    source_url: str
    published: str  # Standardized as string, preserving exact corpus value
    version: int  # Standardized as integer (major version for semver values)
    content: str


class ParserDiagnostic(BaseModel):
    """Diagnostic record for malformed or suspicious lines encountered during parsing."""

    severity: str = "warning"
    line_number: int
    raw_line: str
    message: str


class ParseResult(BaseModel):
    """Complete result of parsing a corpus file."""

    chunks: List[CorpusChunk]
    removed_artifacts_count: int
    diagnostics: List[ParserDiagnostic]


def parse_corpus(corpus_path: Path) -> ParseResult:
    """Parse and validate the raw text corpus without modifying the source file.

    Parameters
    ----------
    corpus_path : Path
        Absolute or relative path to data/corpus.txt.

    Returns
    -------
    ParseResult
        Containing the 75 parsed chunks, count of removed pagination artifacts,
        and any diagnostics emitted.

    Raises
    ------
    CorpusNotFoundError
        If the file does not exist.
    CorpusValidationError
        If chunks are invalid, missing metadata, contain duplicate IDs, or
        unexpected counts.
    """
    path = Path(corpus_path)
    if not path.is_file():
        raise CorpusNotFoundError(
            f"Corpus file not found at: {path}. "
            "Ensure the assignment dataset is placed at data/corpus.txt."
        )

    # Read strictly read-only UTF-8
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    chunks: List[CorpusChunk] = []
    diagnostics: List[ParserDiagnostic] = []
    removed_artifacts_count = 0

    current_header: Optional[tuple[str, str, str]] = None  # (index, title, chunk_id)
    current_meta: dict[str, str] = {}
    current_body: list[str] = []
    chunk_start_line = 0

    def finalize_current_chunk(line_num: int):
        nonlocal current_header, current_meta, current_body, chunk_start_line
        if not current_header:
            return

        body_text = "\n".join(current_body).strip()
        if not body_text:
            raise CorpusValidationError(
                f"Chunk '{current_header[2]}' starting around line {chunk_start_line} "
                "has empty content body."
            )

        missing_keys = REQUIRED_METADATA_KEYS - set(current_meta.keys())
        if missing_keys:
            raise CorpusValidationError(
                f"Chunk '{current_header[2]}' around line {chunk_start_line} "
                f"is missing required metadata fields: {sorted(missing_keys)}"
            )

        raw_version = str(current_meta["version"]).strip()
        try:
            version_int = int(raw_version)
        except ValueError:
            try:
                version_int = int(raw_version.split(".")[0])
            except (ValueError, IndexError):
                raise CorpusValidationError(
                    f"Chunk '{current_header[2]}' has invalid version format: {raw_version}"
                )

        # Ensure header chunk_id matches metadata chunk_id
        if current_header[2] != current_meta["chunk_id"]:
            raise CorpusValidationError(
                f"Header chunk_id '{current_header[2]}' does not match metadata "
                f"chunk_id '{current_meta['chunk_id']}' around line {chunk_start_line}"
            )

        chunk = CorpusChunk(
            chunk_id=current_meta["chunk_id"],
            doc_id=current_meta["doc_id"],
            title=current_meta["title"],
            category=current_meta["category"],
            owner=current_meta["owner"],
            source_url=current_meta["source_url"],
            published=str(current_meta["published"]),  # Preserved as string
            version=version_int,  # Standardized as integer
            content=body_text,
        )
        chunks.append(chunk)

        current_header = None
        current_meta = {}
        current_body = []
        chunk_start_line = line_num

    for line_idx, line in enumerate(lines, 1):
        m_head = HEADER_PATTERN.match(line)
        if m_head:
            finalize_current_chunk(line_idx)
            current_header = (m_head.group(1), m_head.group(2), m_head.group(3))
            chunk_start_line = line_idx
            continue

        if current_header and len(current_meta) < len(REQUIRED_METADATA_KEYS):
            m_meta = META_FIELD_PATTERN.match(line)
            if m_meta:
                key, val = m_meta.group(1), m_meta.group(2).strip()
                if key in REQUIRED_METADATA_KEYS:
                    current_meta[key] = val
                    continue

        # Check for confirmed pagination artifacts to remove
        if CONFIRMED_ARTIFACT_PATTERN.match(line):
            removed_artifacts_count += 1
            continue

        # Check for unexpected variants that should not be silently dropped
        if SUSPECT_ARTIFACT_PATTERN.search(line):
            diag = ParserDiagnostic(
                severity="warning",
                line_number=line_idx,
                raw_line=line,
                message=(
                    f"Line {line_idx} contains suspected pagination artifact with unexpected format: '{line}'. "
                    "Line was retained in content rather than silently discarded."
                ),
            )
            diagnostics.append(diag)
            logger.warning(diag.message)

        if current_header is not None:
            current_body.append(line)

    finalize_current_chunk(len(lines) + 1)

    # Post-parse validation
    if len(chunks) != 75:
        raise CorpusValidationError(
            f"Expected exactly 75 corpus chunks, but parsed {len(chunks)}."
        )

    seen_ids: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen_ids:
            raise CorpusValidationError(f"Duplicate chunk_id detected: {c.chunk_id}")
        seen_ids.add(c.chunk_id)

    return ParseResult(
        chunks=chunks,
        removed_artifacts_count=removed_artifacts_count,
        diagnostics=diagnostics,
    )
