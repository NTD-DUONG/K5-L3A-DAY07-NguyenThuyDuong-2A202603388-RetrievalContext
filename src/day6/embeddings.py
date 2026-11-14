"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Local semantic embedding, pooling token windows without truncating text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        import torch
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        torch.manual_seed(0)
        # Use the already downloaded model; provider failures must remain visible.
        snapshot_path = snapshot_download(
            model_name,
            revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
            local_files_only=True,
        )
        self.model = SentenceTransformer(snapshot_path, local_files_only=True, device="cpu")
        self.model.eval()
        self.dimensions = self.model.get_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length
        self.tokenizer = self.model.tokenizer
        self.window_tokens = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        self.revision = Path(snapshot_path).name
        if not self.revision or self.window_tokens <= 0:
            raise ValueError("Cannot determine model revision or usable token window")
        self.last_stats = {}

    @property
    def configuration(self) -> dict:
        return {
            "model_name": self.model_name, "revision": self.revision,
            "dimensions": self.dimensions, "max_seq_length": self.max_seq_length,
            "window_content_tokens": self.window_tokens, "device": "cpu",
            "document_and_query_processing": "same tokenizer; nonoverlapping token windows; model pooling per window; token-count-weighted mean of window vectors; L2 normalization",
            "prefix": "", "random_seed": 0,
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        windows, owners, weights, lengths = [], [], [], []
        for owner, text in enumerate(texts):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            if not token_ids:
                raise ValueError("Cannot embed empty text")
            lengths.append(len(token_ids))
            encoded = self.tokenizer(
                text, max_length=self.max_seq_length, truncation=True,
                return_overflowing_tokens=True, stride=0,
                return_special_tokens_mask=True, return_attention_mask=True,
            )
            accounted = 0
            for index in range(len(encoded["input_ids"])):
                prepared = {key: encoded[key][index] for key in self.tokenizer.model_input_names if key in encoded}
                assert len(prepared["input_ids"]) <= self.max_seq_length
                weight = sum(1 - flag for flag in encoded["special_tokens_mask"][index])
                accounted += weight
                windows.append(prepared)
                owners.append(owner)
                weights.append(weight)
            assert accounted == len(token_ids), "Tokenizer windows lost or repeated content tokens"

        pooled = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        totals = np.zeros(len(texts), dtype=np.float32)
        for start in range(0, len(windows), 16):
            batch = self.tokenizer.pad(windows[start:start + 16], padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                vectors = self.model(batch)["sentence_embedding"].cpu().numpy()
            for index, vector in enumerate(vectors, start):
                owner, weight = owners[index], weights[index]
                pooled[owner] += vector * weight
                totals[owner] += weight
        if len(texts):
            pooled /= totals[:, None]
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(pooled).all() or (norms <= 0).any():
                raise ValueError("Invalid model output")
            pooled /= norms
        self.last_stats = {
            "texts": len(texts), "content_tokens": sum(lengths),
            "max_content_tokens": max(lengths, default=0),
            "texts_using_multiple_windows": sum(length > self.window_tokens for length in lengths),
            "window_count": len(windows), "truncated_tokens": 0,
        }
        return pooled.tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
