"""
Local Zotero database reader for semantic search.

Provides direct SQLite access to Zotero's local database for faster semantic search
when running in local mode.
"""

import os
import sqlite3
import platform
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .utils import is_local_mode


@dataclass
class ZoteroItem:
    """Represents a Zotero item with text content for semantic search."""
    item_id: int
    key: str
    item_type_id: int
    item_type: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    creators: str | None = None
    fulltext: str | None = None
    fulltext_source: str | None = None  # 'pdf' or 'html'
    notes: str | None = None
    extra: str | None = None
    date_added: str | None = None
    date_modified: str | None = None

    def get_searchable_text(self) -> str:
        """
        Combine all text fields into a single searchable string.

        Returns:
            Combined text content for semantic search indexing.
        """
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.creators:
            parts.append(f"Authors: {self.creators}")

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        if self.extra:
            parts.append(f"Extra: {self.extra}")

        if self.notes:
            parts.append(f"Notes: {self.notes}")

        if self.fulltext:
            # Truncate fulltext to avoid overly long documents
            truncated_fulltext = self.fulltext[:5000] + "..." if len(self.fulltext) > 5000 else self.fulltext
            parts.append(f"Content: {truncated_fulltext}")

        return "\n\n".join(parts)


class LocalZoteroReader:
    """
    Direct SQLite reader for Zotero's local database.

    Provides fast access to item metadata and fulltext for semantic search
    without going through the Zotero API.
    """

    def __init__(
        self,
        db_path: str | None = None,
        pdf_max_pages: int | None = None,
        *,
        docling_device: str | None = None,
        docling_num_threads: int | None = None,
    ):
        """
        Initialize the local database reader.

        Args:
            db_path: Optional path to zotero.sqlite. If None, auto-detect.
        """
        self.db_path = db_path or self._find_zotero_db()
        self._connection: sqlite3.Connection | None = None
        self.pdf_max_pages: int | None = pdf_max_pages
        self.docling_device = docling_device
        self.docling_num_threads = docling_num_threads

    def _get_docling_device(self):
        """Return Docling AcceleratorDevice based on config/env."""
        device_raw = (
            self.docling_device
            or os.getenv("ZOTERO_DOCLING_DEVICE")
            or "auto"
        ).strip().lower()

        try:
            from docling.datamodel.pipeline_options import AcceleratorDevice  # type: ignore
        except ImportError:  # pragma: no cover - import paths differ between docling versions
            from docling.datamodel.accelerator_options import AcceleratorDevice  # type: ignore

        if device_raw in {"cpu"}:
            return AcceleratorDevice.CPU
        if device_raw in {"cuda", "gpu"}:
            return AcceleratorDevice.CUDA
        if device_raw in {"mps"}:
            return AcceleratorDevice.MPS
        return AcceleratorDevice.AUTO

    def _get_docling_num_threads(self) -> int:
        """Return Docling thread count based on config/env."""
        if isinstance(self.docling_num_threads, int) and self.docling_num_threads > 0:
            return self.docling_num_threads

        raw = os.getenv("ZOTERO_DOCLING_NUM_THREADS")
        if raw:
            try:
                value = int(raw)
                if value > 0:
                    return value
            except ValueError:
                pass
        return 4

    def _find_zotero_db(self) -> str:
        """
        Auto-detect the Zotero database location based on OS.

        Returns:
            Path to zotero.sqlite file.

        Raises:
            FileNotFoundError: If database cannot be located.
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
        elif system == "Windows":
            # Try Windows 7+ location first
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
            if not db_path.exists():
                # Fallback to XP/2000 location
                db_path = Path(os.path.expanduser("~/Documents and Settings")) / os.getenv("USERNAME", "") / "Zotero" / "zotero.sqlite"
        else:  # Linux and others
            db_path = Path.home() / "Zotero" / "zotero.sqlite"

        if not db_path.exists():
            raise FileNotFoundError(
                f"Zotero database not found at {db_path}. "
                "Please ensure Zotero is installed and has been run at least once."
            )

        return str(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._connection is None:
            # Open in read-only mode for safety
            uri = f"file:{self.db_path}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _get_storage_dir(self) -> Path:
        """Return the Zotero storage directory path based on database location."""
        # Infer storage directory from database path (same parent directory)
        db_parent = Path(self.db_path).parent
        return db_parent / "storage"

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield tuples (attachment_key, path, content_type) for a parent item."""
        conn = self._get_connection()
        query = (
            """
            SELECT ia.itemID as attachmentItemID,
                   ia.parentItemID as parentItemID,
                   ia.path as path,
                   ia.contentType as contentType,
                   att.key as attachmentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.parentItemID = ?
            """
        )
        for row in conn.execute(query, (parent_item_id,)):
            yield row["attachmentKey"], row["path"], row["contentType"]

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str) -> Path | None:
        """Resolve a Zotero attachment path like 'storage:filename.pdf' to a filesystem path."""
        if not zotero_path:
            return None
        storage_dir = self._get_storage_dir()
        if zotero_path.startswith("storage:"):
            rel = zotero_path.split(":", 1)[1]
            # Handle nested paths if present
            parts = [p for p in rel.split("/") if p]
            return storage_dir / attachment_key / Path(*parts)
        # External links not supported in first pass
        return None

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF using Docling with a page cap to avoid stalls."""
        from .converter import convert_file_to_markdown

        # Determine page cap: config value > env > default (10)
        if isinstance(self.pdf_max_pages, int) and self.pdf_max_pages > 0:
            maxpages = self.pdf_max_pages
        else:
            max_pages_env = os.getenv("ZOTERO_PDF_MAXPAGES")
            try:
                maxpages = int(max_pages_env) if max_pages_env else 10
            except ValueError:
                maxpages = 10

        device = self._get_docling_device()
        num_threads = self._get_docling_num_threads()

        # Prefer configured device; fall back to CPU on GPU/memory failures.
        try:
            return convert_file_to_markdown(
                file_path,
                max_num_pages=maxpages,
                image_processing="drop",
                device=device,
                num_threads=num_threads,
            )
        except Exception as e:
            try:
                # Retry on CPU if non-CPU device fails (common with GPU OOM).
                if getattr(device, "value", str(device)) != "cpu":
                    cpu_device = type(device).CPU
                    return convert_file_to_markdown(
                        file_path,
                        max_num_pages=maxpages,
                        image_processing="drop",
                        device=cpu_device,
                        num_threads=num_threads,
                    )
            except Exception:
                pass
            logger.debug("PDF extraction failed for %s: %s", file_path, e, exc_info=True)
            return ""

    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML using markitdown if available; fallback to stripping tags."""
        # Try markitdown first
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(file_path))
            return result.text_content or ""
        except Exception:
            pass
        # Fallback using a simple parser
        try:
            from bs4 import BeautifulSoup  # type: ignore
            html = file_path.read_text(errors="ignore")
            return BeautifulSoup(html, "html.parser").get_text(" ")
        except Exception:
            return ""

    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from a file based on extension, with fallbacks."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_text_from_pdf(file_path)
        if suffix in {".html", ".htm"}:
            return self._extract_text_from_html(file_path)
        # Try Docling for other common non-plain-text formats
        try:
            from .converter import convert_file_to_markdown
            device = self._get_docling_device()
            num_threads = self._get_docling_num_threads()
            return convert_file_to_markdown(
                file_path,
                image_processing="drop",
                device=device,
                num_threads=num_threads,
            )
        except Exception:
            pass
        # Generic best-effort
        try:
            return file_path.read_text(errors="ignore")
        except Exception:
            return ""

    def _get_fulltext_meta_for_item(self, item_id: int):
        meta = []
        for key, path, ctype in self._iter_parent_attachments(item_id):
            meta.append([key, path, ctype])

        return meta

    def _extract_fulltext_for_item(self, item_id: int) -> tuple[str, str] | None:
        """Attempt to extract fulltext and source from the item's best attachment.

        Preference: use PDF when available; fall back to HTML, then other formats.
        Returns (text, source) where source is derived from the chosen attachment.
        """
        best_pdf: tuple[Path, int] | None = None
        best_html: tuple[Path, int] | None = None
        best_other: tuple[Path, int] | None = None

        for key, path, ctype in self._iter_parent_attachments(item_id):
            resolved = self._resolve_attachment_path(key, path or "")
            if not resolved or not resolved.exists():
                continue
            try:
                size = resolved.stat().st_size
            except OSError:
                size = 0

            if ctype == "application/pdf":
                if best_pdf is None or size > best_pdf[1]:
                    best_pdf = (resolved, size)
            elif (ctype or "").startswith("text/html"):
                if best_html is None or size > best_html[1]:
                    best_html = (resolved, size)
            else:
                if best_other is None or size > best_other[1]:
                    best_other = (resolved, size)

        target = (
            (best_pdf[0] if best_pdf else None)
            or (best_html[0] if best_html else None)
            or (best_other[0] if best_other else None)
        )
        if not target:
            return None
        text = self._extract_text_from_file(target)
        if not text:
            return None
        # Truncate to keep embeddings reasonable
        suffix = target.suffix.lower()
        if suffix == ".pdf":
            source = "pdf"
        elif suffix in {".html", ".htm"}:
            source = "html"
        else:
            source = suffix.lstrip(".") or "file"
        return (text[:10000], source)

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_item_count(self) -> int:
        """
        Get total count of non-attachment items.

        Returns:
            Number of items in the library.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT COUNT(*)
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            """
        )
        return cursor.fetchone()[0]

    def get_items_with_text(self, limit: int | None = None, include_fulltext: bool = False) -> list[ZoteroItem]:
        """
        Get all items with their text content for semantic search.

        Args:
            limit: Optional limit on number of items to return.

        Returns:
            List of ZoteroItem objects with text content.
        """
        conn = self._get_connection()

        # Query to get items with their text content (simplified for now)
        query = """
        SELECT
            i.itemID,
            i.key,
            i.itemTypeID,
            it.typeName as item_type,
            i.dateAdded,
            i.dateModified,
            title_val.value as title,
            abstract_val.value as abstract,
            extra_val.value as extra,
            doi_val.value as doi,
            GROUP_CONCAT(n.note, ' ') as notes,
            GROUP_CONCAT(
                CASE
                    WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                    THEN c.lastName || ', ' || c.firstName
                    WHEN c.lastName IS NOT NULL
                    THEN c.lastName
                    ELSE NULL
                END, '; '
            ) as creators
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

        -- Get title
        LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
        LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

        -- Get abstract
        LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
        LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

        -- Get extra field
        LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
        LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

        -- Get DOI field via fields table
        LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
        LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
        LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

        -- Get notes
        LEFT JOIN itemNotes n ON i.itemID = n.parentItemID OR i.itemID = n.itemID

        -- Get creators
        LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
        LEFT JOIN creators c ON ic.creatorID = c.creatorID

        WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')

        GROUP BY i.itemID, i.key, i.itemTypeID, it.typeName, i.dateAdded, i.dateModified,
                 title_val.value, abstract_val.value, extra_val.value

        ORDER BY i.dateModified DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query)
        items = []

        for row in cursor:
            item = ZoteroItem(
                item_id=row['itemID'],
                key=row['key'],
                item_type_id=row['itemTypeID'],
                item_type=row['item_type'],
                doi=row['doi'],
                title=row['title'],
                abstract=row['abstract'],
                creators=row['creators'],
                fulltext=(res := (self._extract_fulltext_for_item(row['itemID']) if include_fulltext else None)) and res[0],
                fulltext_source=res[1] if include_fulltext and res else None,
                notes=row['notes'],
                extra=row['extra'],
                date_added=row['dateAdded'],
                date_modified=row['dateModified']
            )
            items.append(item)

        return items

    # Public helper to quickly check full text metadata for item
    def get_fulltext_meta_for_item(self, item_id: int) -> tuple[str, str] | None:
        return self._get_fulltext_meta_for_item(item_id)

    # Public helper to extract fulltext on demand for a specific item
    def extract_fulltext_for_item(self, item_id: int) -> tuple[str, str] | None:
        return self._extract_fulltext_for_item(item_id)

    def get_item_by_key(self, key: str) -> ZoteroItem | None:
        """
        Get a specific item by its Zotero key.

        Args:
            key: The Zotero item key.

        Returns:
            ZoteroItem if found, None otherwise.
        """
        items = self.get_items_with_text()
        for item in items:
            if item.key == key:
                return item
        return None

    def search_items_by_text(self, query: str, limit: int = 50) -> list[ZoteroItem]:
        """
        Simple text search through item content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching ZoteroItem objects.
        """
        items = self.get_items_with_text()
        matching_items = []

        query_lower = query.lower()

        for item in items:
            searchable_text = item.get_searchable_text().lower()
            if query_lower in searchable_text:
                matching_items.append(item)
                if len(matching_items) >= limit:
                    break

        return matching_items


def get_local_zotero_reader() -> LocalZoteroReader | None:
    """
    Get a LocalZoteroReader instance if in local mode.

    Returns:
        LocalZoteroReader instance if in local mode and database exists,
        None otherwise.
    """
    if not is_local_mode():
        return None

    try:
        return LocalZoteroReader()
    except FileNotFoundError:
        return None


def is_local_db_available() -> bool:
    """
    Check if local Zotero database is available.

    Returns:
        True if local database can be accessed, False otherwise.
    """
    reader = get_local_zotero_reader()
    if reader:
        reader.close()
        return True
    return False
