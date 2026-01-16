"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage and embedding functions
for semantic search over Zotero libraries.
"""

import json
import os
import sys
import atexit
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from threading import Lock

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Custom OpenAI embedding function for ChromaDB."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            import openai
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")

    def name(self) -> str:
        """Return the name of this embedding function."""
        return "openai"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [data.embedding for data in response.data]


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Gemini embedding function for ChromaDB using google-genai."""

    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        try:
            from google import genai
            from google.genai import types
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                http_options = types.HttpOptions(baseUrl=self.base_url)
                client_kwargs["http_options"] = http_options
            self.client = genai.Client(**client_kwargs)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")

    def name(self) -> str:
        """Return the name of this embedding function."""
        return "gemini"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Gemini API."""
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_document",
                    title="Zotero library document"
                )
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        embedding_config: dict[str, Any] | None = None,
        name_override: str | None = None,
    ):
        self.model_name = model_name
        self.name_override = name_override
        self.embedding_config = embedding_config or {}
        self.batch_size = self._get_int_setting(
            "batch_size",
            env_var="ZOTERO_EMBEDDING_BATCH_SIZE",
            default=32,
        )
        self.chunk_size = self._get_int_setting(
            "chunk_size",
            env_var="ZOTERO_EMBEDDING_CHUNK_SIZE",
            default=None,
        )
        raw_device = os.getenv("ZOTERO_EMBEDDING_DEVICE") or self.embedding_config.get(
            "device"
        )
        self._requested_device: Any = raw_device
        self.device = self._resolve_embedding_device(raw_device)

        self._pool: dict[str, Any] | None = None
        self._pool_lock = Lock()
        self._reported_device: str | None = None
        self._device_report_lock = Lock()
        raw_target_devices = self._get_target_devices()
        self._requested_target_devices: list[str] | None = raw_target_devices
        self._target_devices = self._validate_target_devices(raw_target_devices)
        self._use_multi_process = bool(self._target_devices and len(self._target_devices) > 1)
        self._finalize_multi_process_settings()

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            load_device = "cpu" if self._use_multi_process else self.device
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=load_device,
            )
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers")

        atexit.register(self._stop_pool)

    def name(self) -> str:
        """Return the name of this embedding function."""
        return self.name_override or f"huggingface-{self.model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        texts = list(input)
        if not texts:
            return []

        try:
            embeddings = self._encode(texts)
            return embeddings.tolist()
        except Exception as e:
            if not self._is_oom_error(e):
                raise
            logger.warning("Embedding OOM; retrying with safer settings: %s", e)
            embeddings = self._encode_with_oom_fallback(texts)
            return embeddings.tolist()

    def _get_int_setting(self, key: str, *, env_var: str, default: int | None) -> int | None:
        raw = os.getenv(env_var)
        if raw:
            try:
                parsed = int(raw)
                return parsed if parsed > 0 else default
            except ValueError:
                return default

        value = self.embedding_config.get(key)
        if isinstance(value, int):
            return value if value > 0 else default
        if isinstance(value, str) and value.strip():
            try:
                parsed = int(value)
                return parsed if parsed > 0 else default
            except ValueError:
                return default
        return default

    def _get_target_devices(self) -> list[str] | None:
        devices = os.getenv("ZOTERO_EMBEDDING_DEVICES") or self.embedding_config.get(
            "devices"
        )
        if isinstance(devices, str):
            parsed = [p.strip() for p in devices.split(",") if p.strip()]
            return parsed or None
        if isinstance(devices, list):
            parsed = [str(p).strip() for p in devices if str(p).strip()]
            return parsed or None

        gpu_ids = os.getenv("ZOTERO_EMBEDDING_GPU_IDS") or self.embedding_config.get(
            "gpu_ids"
        )
        if isinstance(gpu_ids, str) and gpu_ids.strip():
            parsed = [p.strip() for p in gpu_ids.split(",") if p.strip()]
            return [f"cuda:{p}" for p in parsed] or None
        if isinstance(gpu_ids, list):
            parsed = [str(p).strip() for p in gpu_ids if str(p).strip()]
            return [f"cuda:{p}" for p in parsed] or None
        return None

    def _validate_target_devices(self, devices: list[str] | None) -> list[str] | None:
        if not devices:
            return None

        validated: list[str] = []
        for raw in devices:
            if not isinstance(raw, str):
                continue
            dev = raw.strip().lower()
            if not dev:
                continue

            if dev.isdigit():
                dev = f"cuda:{dev}"

            if dev in {"cpu"}:
                validated.append("cpu")
                continue

            if dev in {"cuda", "gpu"} or dev.startswith("cuda:"):
                resolved = self._validated_cuda_device(dev, emit_warnings=False)
                if resolved.startswith("cuda"):
                    validated.append(resolved)
                continue

            if dev.startswith("mps"):
                resolved = self._validated_mps_device(dev, emit_warnings=False)
                if resolved == "mps":
                    validated.append("mps")
                continue

            # Unknown/unsupported device strings: keep as-is (lets upstream handle errors).
            validated.append(dev)

        validated = [d for i, d in enumerate(validated) if d and d not in validated[:i]]
        return validated or None

    def _finalize_multi_process_settings(self) -> None:
        # If the user requested explicit target devices but none validated, be explicit
        # about falling back and avoid starting a misconfigured pool.
        if self._requested_target_devices and not self._target_devices:
            try:
                sys.stderr.write(
                    "Warning: requested embedding devices "
                    f"{', '.join(self._requested_target_devices)} but none are available; "
                    f"falling back to single-process on {self.device}.\n"
                )
            except Exception:
                pass
            self._use_multi_process = False
            self._target_devices = None
            return

        # If a single device is specified via the multi-device setting, treat it as
        # an alias for `device` so callers can pass `--embedding-devices cuda:1`.
        if self._target_devices and len(self._target_devices) == 1:
            self.device = self._target_devices[0]
            self._target_devices = None
            self._use_multi_process = False
            return

        # Multi-process is only used when multiple target devices are specified.
        self._use_multi_process = bool(self._target_devices and len(self._target_devices) > 1)

    def _resolve_embedding_device(self, raw_device: Any) -> str | None:
        if not isinstance(raw_device, str):
            device = None
        else:
            device = raw_device.strip().lower()

        if device in {None, "", "auto"}:
            return self._auto_device()

        if device == "gpu":
            device = "cuda"

        if isinstance(device, str) and device.isdigit():
            device = f"cuda:{device}"

        if device.startswith("cuda"):
            return self._validated_cuda_device(device)

        if device.startswith("mps"):
            return self._validated_mps_device(device)

        return device

    def _auto_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    idx = torch.cuda.current_device()
                    return f"cuda:{idx}"
                except Exception:
                    return "cuda"
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _validated_cuda_device(self, device: str, *, emit_warnings: bool = True) -> str:
        try:
            import torch
        except Exception as e:
            if emit_warnings:
                try:
                    sys.stderr.write(
                        f"Warning: requested embedding device '{device}' but torch is unavailable; falling back to CPU ({e}).\n"
                    )
                except Exception:
                    pass
            return "cpu"

        if not torch.cuda.is_available():
            if emit_warnings:
                try:
                    sys.stderr.write(
                        f"Warning: requested embedding device '{device}' but CUDA is unavailable; falling back to CPU.\n"
                    )
                except Exception:
                    pass
            return "cpu"

        count = 0
        try:
            count = int(torch.cuda.device_count())
        except Exception:
            count = 0
        if count <= 0:
            if emit_warnings:
                try:
                    sys.stderr.write(
                        f"Warning: requested embedding device '{device}' but no CUDA devices were found; falling back to CPU.\n"
                    )
                except Exception:
                    pass
            return "cpu"

        try:
            current_idx = int(torch.cuda.current_device())
        except Exception:
            current_idx = 0

        index = current_idx
        if ":" in device:
            _, raw_index = device.split(":", 1)
            try:
                parsed = int(raw_index)
                if 0 <= parsed < count:
                    index = parsed
                else:
                    raise ValueError(f"CUDA device index {parsed} out of range (count={count})")
            except Exception as e:
                if emit_warnings:
                    try:
                        sys.stderr.write(
                            f"Warning: requested embedding device '{device}' is invalid; using cuda:{current_idx} instead ({e}).\n"
                        )
                    except Exception:
                        pass
                index = current_idx

        return f"cuda:{index}"

    def _validated_mps_device(self, device: str, *, emit_warnings: bool = True) -> str:
        try:
            import torch

            if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():  # type: ignore[attr-defined]
                raise RuntimeError("torch.backends.mps.is_available() is False")
            return "mps"
        except Exception as e:
            if emit_warnings:
                try:
                    sys.stderr.write(
                        f"Warning: requested embedding device '{device}' but MPS is unavailable; falling back to CPU ({e}).\n"
                    )
                except Exception:
                    pass
            return "cpu"

    def _ensure_pool(self) -> dict[str, Any] | None:
        if not self._use_multi_process:
            return None
        with self._pool_lock:
            if self._pool is None:
                try:
                    if self._target_devices:
                        self._pool = self.model.start_multi_process_pool(
                            target_devices=self._target_devices
                        )
                    else:
                        self._pool = self.model.start_multi_process_pool()
                except Exception as e:
                    logger.warning("Failed to start embedding multi-process pool: %s", e)
                    try:
                        sys.stderr.write(
                            f"Warning: failed to start embedding multi-process pool; falling back to single-process on {self.device} ({e}).\n"
                        )
                    except Exception:
                        pass
                    self._pool = None
        return self._pool

    def get_runtime_device_description(
        self, *, ensure_pool: bool = False, mark_reported: bool = False
    ) -> str:
        """
        Return a human-friendly description of the embedding runtime device(s).

        For multi-process embeddings this can optionally start the pool to resolve
        the actual target devices.
        """
        pool = self._ensure_pool() if (ensure_pool and self._use_multi_process) else self._pool
        if pool is not None:
            target_devices = pool.get("target_devices") or self._target_devices
            if target_devices:
                desc = f"multi-process ({', '.join(target_devices)})"
            else:
                desc = "multi-process (auto)"
        else:
            # If multi-process is enabled but the pool isn't running yet, report the configured
            # target devices so callers can see what will be used once embedding starts.
            if self._use_multi_process:
                if self._target_devices:
                    desc = f"multi-process ({', '.join(self._target_devices)})"
                else:
                    desc = "multi-process (auto)"
            else:
                desc = str(self._get_model_device_string() or self.device or "unknown")

        if mark_reported:
            with self._device_report_lock:
                self._reported_device = desc

        return desc

    def _get_model_device_string(self) -> str | None:
        try:
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                return self._format_torch_device(model_device)
        except Exception:
            pass
        return None

    def _format_torch_device(self, device_obj: Any) -> str:
        try:
            import torch

            if isinstance(device_obj, torch.device) and device_obj.type == "cuda":
                idx = device_obj.index
                if idx is None:
                    try:
                        idx = torch.cuda.current_device()
                    except Exception:
                        idx = None
                if idx is not None:
                    return f"cuda:{idx}"
        except Exception:
            pass
        return str(device_obj)

    def _stop_pool(self) -> None:
        pool = self._pool
        if pool is None:
            return
        with self._pool_lock:
            pool = self._pool
            self._pool = None
        try:
            self.model.stop_multi_process_pool(pool)
        except Exception:
            pass

    def _encode(self, texts: list[str]):
        pool = self._ensure_pool()
        self._report_runtime_device(pool)
        if pool is not None:
            return self.model.encode_multi_process(
                texts,
                pool,
                batch_size=self.batch_size or 32,
                chunk_size=self.chunk_size,
                show_progress_bar=False,
            )
        return self.model.encode(
            texts,
            batch_size=self.batch_size or 32,
            chunk_size=self.chunk_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
        )

    def _report_runtime_device(self, pool: dict[str, Any] | None) -> None:
        if pool is not None:
            target_devices = pool.get("target_devices") or self._target_devices
            if target_devices:
                reported = f"multi-process ({', '.join(target_devices)})"
                line = f"Embedding devices: {', '.join(target_devices)}\n"
            else:
                reported = "multi-process (auto)"
                line = "Embedding devices: (multi-process auto)\n"
        else:
            if self._use_multi_process:
                reported = str(self.device or "unknown")
            else:
                reported = str(self._get_model_device_string() or self.device or "unknown")
            line = f"Embedding device: {self._describe_single_device(reported)}\n"

        with self._device_report_lock:
            if self._reported_device == reported:
                return
            self._reported_device = reported

        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass

    def _describe_single_device(self, device: str) -> str:
        if not isinstance(device, str):
            return str(device)
        dev = device.strip()
        if not dev.startswith("cuda"):
            return dev
        return self._describe_cuda_device(dev)

    def _describe_cuda_device(self, device: str) -> str:
        try:
            import torch

            idx: int | None = None
            if ":" in device:
                _, raw_idx = device.split(":", 1)
                try:
                    idx = int(raw_idx)
                except ValueError:
                    idx = None
            if idx is None:
                try:
                    idx = int(torch.cuda.current_device())
                except Exception:
                    idx = None
            if idx is None:
                return device

            name = torch.cuda.get_device_name(idx)
            total = None
            try:
                props = torch.cuda.get_device_properties(idx)
                total = float(props.total_memory) / (1024.0 * 1024.0)
            except Exception:
                total = None

            alloc = None
            reserved = None
            try:
                alloc = float(torch.cuda.memory_allocated(idx)) / (1024.0 * 1024.0)
                reserved = float(torch.cuda.memory_reserved(idx)) / (1024.0 * 1024.0)
            except Exception:
                alloc = None
                reserved = None

            extra_parts = [name]
            if alloc is not None and reserved is not None:
                if total is not None:
                    extra_parts.append(
                        f"mem {alloc:.0f}/{reserved:.0f}/{total:.0f} MiB"
                    )
                else:
                    extra_parts.append(f"mem {alloc:.0f}/{reserved:.0f} MiB")
            return f"{device} ({'; '.join(extra_parts)})"
        except Exception:
            return device

    def _encode_with_oom_fallback(self, texts: list[str]):
        self._maybe_empty_cuda_cache()
        self._stop_pool()

        last_exc: Exception | None = None
        device = self.device
        tried_accelerator = isinstance(device, str) and device.strip().lower() not in {
            "cpu"
        }
        if tried_accelerator:
            for bs in self._candidate_batch_sizes(self.batch_size or 32):
                try:
                    return self.model.encode(
                        texts,
                        batch_size=bs,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=device,
                    )
                except Exception as e:
                    last_exc = e
                    if not self._is_oom_error(e):
                        raise
                    self._maybe_empty_cuda_cache()

        if tried_accelerator:
            try:
                sys.stderr.write(
                    f"Warning: embedding ran out of memory on {device}; falling back to CPU.\n"
                )
            except Exception:
                pass

        # Final fallback: CPU with small batch size (and keep CPU for the rest of the run).
        self._use_multi_process = False
        self._target_devices = None
        self.device = "cpu"
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self._report_runtime_device(None)
        for bs in (8, 4, 2, 1):
            try:
                return self.model.encode(
                    texts,
                    batch_size=bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device="cpu",
                )
            except Exception as e:
                last_exc = e
                if not self._is_oom_error(e):
                    raise
        if last_exc:
            raise last_exc
        raise RuntimeError("Failed to encode embeddings after OOM fallbacks")

    def _candidate_batch_sizes(self, start: int) -> list[int]:
        start = max(1, int(start))
        candidates = []
        bs = start
        while bs >= 1:
            candidates.append(bs)
            if bs == 1:
                break
            bs = max(1, bs // 2)
        return candidates

    def _maybe_empty_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _is_oom_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg or "cudnn" in msg


class ChromaClient:
    """ChromaDB client for Zotero semantic search."""

    def __init__(self,
                 collection_name: str = "zotero_library",
                 persist_directory: str | None = None,
                 embedding_model: str = "default",
                 embedding_config: dict[str, Any] | None = None):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini', 'qwen', 'embeddinggemma', or HuggingFace model name)
            embedding_config: Configuration for the embedding model
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}

        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")

        self.persist_directory = persist_directory

        # Initialize ChromaDB client with stdout suppression
        with suppress_stdout():
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Set up embedding function
            self.embedding_function = self._create_embedding_function()

            # Get or create collection with embedding function handling
            try:
                # Try to get existing collection first
                self.collection = self.client.get_collection(name=self.collection_name)

                # Check if embedding functions are compatible
                existing_ef = getattr(self.collection, '_embedding_function', None)
                if existing_ef is not None:
                    existing_name = getattr(existing_ef, 'name', lambda: 'default')()
                    new_name = getattr(self.embedding_function, 'name', lambda: 'default')()

                    if existing_name != new_name:
                        # Log to stderr instead of letting ChromaDB print to stdout
                        sys.stderr.write(f"ChromaDB: Collection exists with different embedding function: {existing_name} vs {new_name}\n")
                        # Use the existing collection's embedding function to avoid conflicts
                        self.embedding_function = existing_ef

            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )

    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration."""
        if self.embedding_model == "openai":
            model_name = self.embedding_config.get("model_name", "text-embedding-3-small")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)

        elif self.embedding_model == "gemini":
            model_name = self.embedding_config.get("model_name", "models/text-embedding-004")
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_config.get("base_url")
            return GeminiEmbeddingFunction(model_name=model_name, api_key=api_key, base_url=base_url)

        elif self.embedding_model == "qwen":
            model_name = self.embedding_config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
            return HuggingFaceEmbeddingFunction(
                model_name=model_name, embedding_config=self.embedding_config
            )

        elif self.embedding_model == "embeddinggemma":
            model_name = self.embedding_config.get("model_name", "google/embeddinggemma-300m")
            return HuggingFaceEmbeddingFunction(
                model_name=model_name, embedding_config=self.embedding_config
            )

        elif self.embedding_model not in ["default", "openai", "gemini"]:
            # Treat any other value as a HuggingFace model name
            return HuggingFaceEmbeddingFunction(
                model_name=self.embedding_model, embedding_config=self.embedding_config
            )

        else:
            # Use a sentence-transformers embedding function compatible with Chroma's default.
            return HuggingFaceEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embedding_config=self.embedding_config,
                name_override="default",
            )

    def add_documents(self,
                     documents: list[str],
                     metadatas: list[dict[str, Any]],
                     ids: list[str]) -> None:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def upsert_documents(self,
                        documents: list[str],
                        metadatas: list[dict[str, Any]],
                        ids: list[str]) -> None:
        """
        Upsert (update or insert) documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            logger.error(f"Error upserting documents to ChromaDB: {e}")
            raise

    def search(self,
               query_texts: list[str],
               n_results: int = 10,
               where: dict[str, Any] | None = None,
               where_document: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            Search results from ChromaDB
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(f"Semantic search returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            raise

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise

    def get_embedding_description(self) -> str:
        """Return a human-friendly description of the active embedding model."""
        collection = getattr(self, "collection", None)
        ef = getattr(collection, "_embedding_function", None) or getattr(self, "embedding_function", None)
        if ef is None:
            return "unknown"

        try:
            if isinstance(ef, OpenAIEmbeddingFunction):
                return f"openai ({ef.model_name})"
            if isinstance(ef, GeminiEmbeddingFunction):
                return f"gemini ({ef.model_name})"
            if isinstance(ef, HuggingFaceEmbeddingFunction):
                if getattr(ef, "name_override", None) == "default":
                    return "default (all-MiniLM-L6-v2)"
                return f"huggingface ({ef.model_name})"
        except Exception:
            pass

        try:
            ef_name = ef.name() if hasattr(ef, "name") else None
        except Exception:
            ef_name = None

        if ef_name == "default":
            return "default (all-MiniLM-L6-v2)"

        model_name = getattr(ef, "model_name", None)
        if ef_name and model_name:
            return f"{ef_name} ({model_name})"
        if ef_name:
            return str(ef_name)
        if model_name:
            return f"{type(ef).__name__} ({model_name})"
        return type(ef).__name__

    def get_embedding_device_description(
        self, *, ensure_pool: bool = False, mark_reported: bool = False
    ) -> str:
        """Return a human-friendly description of the embedding runtime device."""
        collection = getattr(self, "collection", None)
        ef = getattr(collection, "_embedding_function", None) or getattr(self, "embedding_function", None)
        if ef is None:
            return "unknown"

        try:
            if isinstance(ef, HuggingFaceEmbeddingFunction):
                return ef.get_runtime_device_description(
                    ensure_pool=ensure_pool, mark_reported=mark_reported
                )
            if isinstance(ef, OpenAIEmbeddingFunction):
                return "remote (openai)"
            if isinstance(ef, GeminiEmbeddingFunction):
                return "remote (gemini)"
        except Exception:
            pass

        return "unknown"

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "embedding_description": self.get_embedding_description(),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "embedding_description": self.get_embedding_description(),
                "persist_directory": self.persist_directory,
                "error": str(e)
            }

    def reset_collection(self) -> None:
        """Reset (clear) the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Reset ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get metadata for a document if it exists.

        Args:
            doc_id: Document ID to look up

        Returns:
            Metadata dictionary if document exists, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["metadatas"])
            if result['ids'] and result['metadatas']:
                return result['metadatas'][0]
            return None
        except Exception:
            return None

    def get_documents_metadata(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """
        Batch metadata lookup for many documents.

        Args:
            doc_ids: Document IDs to look up

        Returns:
            Mapping of doc_id -> metadata for documents that exist.
        """
        if not doc_ids:
            return {}

        try:
            result = self.collection.get(ids=doc_ids, include=["metadatas"])
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            out: dict[str, dict[str, Any]] = {}
            for doc_id, meta in zip(ids, metadatas):
                if doc_id:
                    out[str(doc_id)] = meta or {}
            return out
        except Exception:
            return {}


def create_chroma_client(config_path: str | None = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured ChromaClient instance
    """
    # Default configuration
    config = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {}
    }

    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Load configuration from environment variables
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model:
        config["embedding_model"] = env_embedding_model

    # Set up embedding config from environment
    if config["embedding_model"] == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_api_key:
            config["embedding_config"] = {
                "api_key": openai_api_key,
                "model_name": openai_model
            }
            if openai_base_url:
                config["embedding_config"]["base_url"] = openai_base_url

    elif config["embedding_model"] == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        if gemini_api_key:
            config["embedding_config"] = {
                "api_key": gemini_api_key,
                "model_name": gemini_model
            }
            if gemini_base_url:
                config["embedding_config"]["base_url"] = gemini_base_url

    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )
