import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from openai import OpenAI, OpenAIError
from werkzeug.utils import secure_filename

from maintenance_assistant import (
    DocumentProcessingError,
    DocumentChunk,
    answer_question,
    chunk_documents,
    embed_texts,
    extract_documents,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_MANUAL = ROOT / "data" / "sample_manual.md"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class IndexedDocuments:
    client: OpenAI
    chunks: list[DocumentChunk]
    vectors: list[list[float]]


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
indexes: dict[str, IndexedDocuments] = {}


def get_current_index() -> IndexedDocuments | None:
    index_id = session.get("index_id")
    if not isinstance(index_id, str):
        return None
    return indexes.get(index_id)


@app.get("/")
def home():
    return render_template("index.html", index_ready=get_current_index() is not None)


@app.post("/index")
def index_documents():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        flash("Set OPENAI_API_KEY in your environment or a local .env file.", "error")
        return redirect(url_for("home"))

    sources = [(SAMPLE_MANUAL.name, SAMPLE_MANUAL.read_bytes())]
    for uploaded in request.files.getlist("documents"):
        if not uploaded.filename:
            continue
        filename = secure_filename(uploaded.filename)
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            flash(
                f"Unsupported file type for '{filename}'. Use PDF, TXT, or MD.",
                "error",
            )
            return redirect(url_for("home"))
        sources.append((filename, uploaded.read()))

    try:
        documents = extract_documents(sources)
        chunks = chunk_documents(documents)
        if not chunks:
            raise DocumentProcessingError(
                "No readable text was found in the selected documents."
            )

        client = OpenAI(api_key=api_key)
        vectors = embed_texts(
            client,
            [chunk.text for chunk in chunks],
            EMBEDDING_MODEL,
        )
        previous_index_id = session.get("index_id")
        if isinstance(previous_index_id, str):
            indexes.pop(previous_index_id, None)
        index_id = uuid.uuid4().hex
        indexes[index_id] = IndexedDocuments(client, chunks, vectors)
        session["index_id"] = index_id
        flash(
            f"Indexed {len(chunks)} passages from {len(sources)} document(s).",
            "success",
        )
    except (DocumentProcessingError, OpenAIError, OSError, ValueError) as error:
        logger.exception("Could not index maintenance documents")
        flash(f"Could not index documents: {error}", "error")

    return redirect(url_for("home"))


@app.errorhandler(413)
def request_too_large(_error):
    return (
        render_template(
            "index.html",
            index_ready=get_current_index() is not None,
            error="The upload is too large. Keep the combined file size under 16 MB.",
        ),
        413,
    )


@app.post("/ask")
def ask():
    index = get_current_index()
    if index is None:
        session.pop("index_id", None)
        flash("Index the documents before asking a question.", "error")
        return redirect(url_for("home"))

    question = request.form.get("question", "").strip()
    if not question:
        flash("Enter a question before asking.", "error")
        return redirect(url_for("home"))

    try:
        answer, sources = answer_question(
            client=index.client,
            chunks=index.chunks,
            vectors=index.vectors,
            question=question,
            embedding_model=EMBEDDING_MODEL,
            chat_model=CHAT_MODEL,
        )
        return render_template(
            "index.html",
            index_ready=True,
            question=question,
            answer=answer,
            sources=sources,
        )
    except OpenAIError as error:
        logger.exception("OpenAI request failed")
        flash(f"The AI service request failed: {error}", "error")
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
