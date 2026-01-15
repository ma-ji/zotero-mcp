"""Multiprocessing helpers for parallel full-text extraction from the local Zotero DB."""

from __future__ import annotations

import os
import logging
from multiprocessing import current_process

from .local_db import LocalZoteroReader

logger = logging.getLogger(__name__)

_reader: LocalZoteroReader | None = None


def init_fulltext_worker(
    db_path: str | None,
    pdf_max_pages: int | None,
    docling_device: str | None,
    docling_num_threads: int | None,
    gpu_ids: list[str] | None,
) -> None:
    """Initializer for extraction workers.

    If gpu_ids is provided, each worker process will be pinned to a single GPU
    by setting CUDA_VISIBLE_DEVICES to a single GPU id.
    """
    global _reader

    if gpu_ids:
        try:
            ident = getattr(current_process(), "_identity", None) or ()
            worker_index = (ident[0] - 1) if ident else 0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                gpu_ids[worker_index % len(gpu_ids)]
            )
        except Exception as e:
            logger.debug("Failed to set CUDA_VISIBLE_DEVICES: %s", e)

    _reader = LocalZoteroReader(
        db_path=db_path,
        pdf_max_pages=pdf_max_pages,
        docling_device=docling_device,
        docling_num_threads=docling_num_threads,
    )


def extract_fulltext_for_item(item_id: int) -> tuple[int, str | None, str | None]:
    """Return (item_id, fulltext, source) for a Zotero item ID."""
    global _reader

    if _reader is None:
        _reader = LocalZoteroReader()

    try:
        result = _reader.extract_fulltext_for_item(item_id)
        if not result:
            return (item_id, None, None)
        if isinstance(result, tuple) and len(result) == 2:
            return (item_id, result[0], result[1])
        return (item_id, str(result), None)
    except Exception as e:
        logger.debug("Fulltext extraction failed for item %s: %s", item_id, e)
        return (item_id, None, None)
