from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, Sequence

from openai import OpenAI
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentProcessingError(ValueError):
    """Raised when source documents cannot be prepared for retrieval."""


@dataclass(frozen=True)
class Document:
    source: str
    page: int | None
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    page: int | None
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    page: int | None
    text: str
    score: float


def extract_documents(files: Iterable[tuple[str, bytes]]) -> list[Document]:
    documents: list[Document] = []
    for filename, content in files:
        extension = filename.lower().rsplit(".", maxsplit=1)[-1]
        if extension == "pdf":
            try:
                reader = PdfReader(BytesIO(content))
            except PdfReadError as error:
                raise DocumentProcessingError(
                    f"Could not read PDF '{filename}': {error}"
                ) from error
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(Document(filename, page_number, text))
        elif extension in {"txt", "md"}:
            text = content.decode("utf-8", errors="replace").strip()
            if text:
                documents.append(Document(filename, None, text))
        else:
            raise DocumentProcessingError(
                f"Unsupported file type for '{filename}'. Use PDF, TXT, or MD."
            )
    return documents


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[DocumentChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for document in documents:
        text = " ".join(document.text.split())
        for start in range(0, len(text), step):
            part = text[start : start + chunk_size].strip()
            if part:
                chunks.append(
                    DocumentChunk(document.source, document.page, part)
                )
            if start + chunk_size >= len(text):
                break
    return chunks


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    model: str,
) -> list[list[float]]:
    if not texts:
        return []

    vectors: list[list[float] | None] = [None] * len(texts)
    batch_size = 100
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        response = client.embeddings.create(model=model, input=list(batch))
        for item in response.data:
            vectors[batch_start + item.index] = item.embedding

    if any(vector is None for vector in vectors):
        raise RuntimeError("The embedding service returned an incomplete response.")
    return [vector for vector in vectors if vector is not None]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have equal dimensions.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def retrieve_chunks(
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    top_k: int = 4,
    minimum_similarity: float = 0.22,
) -> list[RetrievedChunk]:
    if len(chunks) != len(vectors):
        raise ValueError("Each document passage must have one embedding.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    scored = [
        RetrievedChunk(
            source=chunk.source,
            page=chunk.page,
            text=chunk.text,
            score=cosine_similarity(vector, query_vector),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    scored.sort(key=lambda result: result.score, reverse=True)
    return [
        result
        for result in scored[:top_k]
        if result.score >= minimum_similarity
    ]


def answer_question(
    client: OpenAI,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[Sequence[float]],
    question: str,
    embedding_model: str,
    chat_model: str,
    minimum_similarity: float = 0.22,
) -> tuple[str, list[RetrievedChunk]]:
    if not question.strip():
        raise ValueError("Enter a question before asking.")

    query_vector = embed_texts(client, [question.strip()], embedding_model)[0]
    sources = retrieve_chunks(
        chunks,
        vectors,
        query_vector,
        minimum_similarity=minimum_similarity,
    )
    if not sources:
        return (
            "I couldn't find supporting information in the indexed documents. "
            "Please consult the equipment manual or a qualified technician.",
            [],
        )

    source_text = "\n\n".join(
        f"[S{index}] Source: {source.source}"
        f"{f', page {source.page}' if source.page is not None else ''}\n"
        f"{source.text}"
        for index, source in enumerate(sources, start=1)
    )
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious manufacturing maintenance information "
                    "assistant. Answer only from the supplied source passages; "
                    "do not add outside knowledge or infer missing procedures. "
                    "Cite factual claims with the exact source labels, such as "
                    "[S1]. If the passages do not answer the question, say that "
                    "the documents do not provide the information. Never advise "
                    "bypassing guards, interlocks, lockout/tagout, or site safety "
                    "procedures. Do not present yourself as a substitute for a "
                    "qualified technician. Be concise and distinguish source "
                    "facts from any limitation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question.strip()}\n\n"
                    f"Source passages:\n{source_text}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The chat model returned an empty answer.")
    return answer, sources
