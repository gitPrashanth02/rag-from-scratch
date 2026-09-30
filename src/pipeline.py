"""
Top-level RAG pipeline.

This is the single object the Flask app talks to. It owns one Generator
and lazily builds a Retriever for the DPR embedder.
"""

import logging
import time
from typing import Literal

from src.embeddings_dpr import DPREmbedder
from src.generation import Generator
from src.retrieval import Retriever
from src import config

logger = logging.getLogger(__name__)

EmbedderName = Literal["dpr"]


class RagPipeline:
    def __init__(self):
        self._embedders = {}
        self._retrievers = {}
        self.generator = None

    def _get_embedder(self, name: EmbedderName):
        if name not in self._embedders:
            logger.info("Loading embedder backend: %s", name)
            if name == "dpr":
                self._embedders[name] = DPREmbedder()
            else:
                raise ValueError(f"Unknown embedder backend: {name}")
        return self._embedders[name]

    def _get_retriever(self, name: EmbedderName) -> Retriever:
        if name not in self._retrievers:
            embedder = self._get_embedder(name)
            self._retrievers[name] = Retriever(embedder)
        return self._retrievers[name]

    def _get_generator(self) -> Generator:
        if self.generator is None:
            logger.info("Loading generator model")
            self.generator = Generator()
        return self.generator

    def warm_up(self, embedders: tuple[EmbedderName, ...] = ("dpr",)) -> None:
        """Preload all models up front so the first HTTP request isn't slow."""
        for name in embedders:
            self._get_retriever(name)
        self._get_generator()

    def ask(self, question: str, embedder_name: EmbedderName = "dpr", k: int = config.DEFAULT_TOP_K) -> dict:
        """
        Retrieve relevant passages and generate an answer grounded in them.

        Returns a dict shaped for direct JSON serialization by the Flask API.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")

        generator = self._get_generator()
        retriever = self._get_retriever(embedder_name)

        t0 = time.time()
        retrieved_chunks, scores = retriever.retrieve(question, k=k)
        retrieval_time = time.time() - t0

        t0 = time.time()
        answer_with = generator.answer_with_retrieval(question, retrieved_chunks)
        with_retrieval_time = time.time() - t0

        return {
            "question": question,
            "embedder": embedder_name,
            "retrieved_chunks": [
                {"text": chunk, "distance": round(score, 4)}
                for chunk, score in zip(retrieved_chunks, scores)
            ],
            "answer_with_retrieval": answer_with,
            "timing_seconds": {
                "retrieval": round(retrieval_time, 3),
                "generation": round(with_retrieval_time, 3),
            },
        }
