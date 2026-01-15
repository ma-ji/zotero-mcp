"""Helpers for converting attachments into text/markdown using Docling."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import local
from typing import Literal

logger = logging.getLogger(__name__)
_converter_local = local()

ImageProcessing = Literal["embed", "placeholder", "drop"]


def get_pipeline_options(
    *,
    force_full_page_ocr: bool,
    do_picture_description: bool,
    image_resolution_scale: float,
    device: AcceleratorDevice | None = None,
    num_threads: int = 4,
) -> PdfPipelineOptions:
    """Return Docling PdfPipelineOptions using the same defaults as `ref/converter.py`."""

    try:
        from docling.datamodel.pipeline_options import (  # type: ignore
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
            TableFormerMode,
        )
    except ImportError:  # pragma: no cover - import paths differ between docling versions
        from docling.datamodel.accelerator_options import (  # type: ignore
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.pipeline_options import (  # type: ignore
            PdfPipelineOptions,
            TableFormerMode,
        )

    if device is None:
        device = AcceleratorDevice.AUTO

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = do_picture_description
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_code_enrichment = True
    pipeline_options.ocr_options.force_full_page_ocr = force_full_page_ocr
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.images_scale = image_resolution_scale

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads, device=device
    )
    return pipeline_options


def convert_file_to_markdown(
    file_path: str | Path,
    *,
    force_full_page_ocr: bool = False,
    do_picture_description: bool = False,
    image_resolution_scale: float = 2.0,
    image_processing: ImageProcessing = "placeholder",
    max_num_pages: int | None = None,
    max_file_size: int | None = None,
    device: AcceleratorDevice | None = None,
    num_threads: int = 4,
) -> str:
    """Convert a local file into Markdown using Docling.

    Docling supports multiple formats (PDF, DOCX, PPTX, XLSX, HTML, images, ...).
    This function configures PDF conversion with the same defaults as
    `ref/converter.py` and relies on Docling defaults for other formats.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return _render_markdown(
        path,
        force_full_page_ocr=force_full_page_ocr,
        do_picture_description=do_picture_description,
        image_resolution_scale=image_resolution_scale,
        image_processing=image_processing,
        max_num_pages=max_num_pages,
        max_file_size=max_file_size,
        device=device,
        num_threads=num_threads,
    )


def _render_markdown(
    file_path: Path,
    *,
    force_full_page_ocr: bool,
    do_picture_description: bool,
    image_resolution_scale: float,
    image_processing: ImageProcessing,
    max_num_pages: int | None,
    max_file_size: int | None,
    device: AcceleratorDevice | None,
    num_threads: int,
) -> str:
    from docling.datamodel.base_models import InputFormat  # type: ignore
    from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore

    try:
        from docling.datamodel.pipeline_options import AcceleratorDevice  # type: ignore
    except ImportError:  # pragma: no cover - import paths differ between docling versions
        from docling.datamodel.accelerator_options import AcceleratorDevice  # type: ignore

    try:
        from docling_core.types.doc import ImageRefMode  # type: ignore
    except ImportError:  # pragma: no cover - older docling-core import path
        from docling_core.types.doc.base import ImageRefMode  # type: ignore

    if device is None:
        device = AcceleratorDevice.AUTO

    cache_key = (
        force_full_page_ocr,
        do_picture_description,
        image_resolution_scale,
        device,
        num_threads,
    )
    cached_key = getattr(_converter_local, "key", None)
    converter = getattr(_converter_local, "converter", None)

    if converter is None or cached_key != cache_key:
        pipeline_options = get_pipeline_options(
            force_full_page_ocr=force_full_page_ocr,
            do_picture_description=do_picture_description,
            image_resolution_scale=image_resolution_scale,
            device=device,
            num_threads=num_threads,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        _converter_local.key = cache_key
        _converter_local.converter = converter

    convert_kwargs: dict[str, int] = {}
    if isinstance(max_num_pages, int) and max_num_pages > 0:
        convert_kwargs["max_num_pages"] = max_num_pages
    if isinstance(max_file_size, int) and max_file_size > 0:
        convert_kwargs["max_file_size"] = max_file_size

    result = converter.convert(str(file_path), **convert_kwargs)

    if result.document is None:
        msg = f"Docling conversion failed for {file_path}: {getattr(result, 'status', None)}"
        raise RuntimeError(msg)

    if image_processing == "embed":
        image_mode = ImageRefMode.EMBEDDED
        image_placeholder = "<!-- image -->"
    elif image_processing == "placeholder":
        image_mode = ImageRefMode.PLACEHOLDER
        image_placeholder = "<!-- image -->"
    elif image_processing == "drop":
        image_mode = ImageRefMode.PLACEHOLDER
        image_placeholder = ""
    else:  # pragma: no cover - validated by ImageProcessing literal
        image_mode = ImageRefMode.PLACEHOLDER
        image_placeholder = "<!-- image -->"

    return result.document.export_to_markdown(
        image_placeholder=image_placeholder,
        image_mode=image_mode,
        page_break_placeholder="\n\n--- Page Break ---\n\n",
    )

