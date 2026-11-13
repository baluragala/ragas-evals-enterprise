"""Load Aurora Bank's policy documents and index them for retrieval (LangChain 1.x).

Two chunking strategies are offered so you can *measure* the difference with Ragas:

* header-aware (default): split on Markdown headings first, then by size, and prefix every chunk
  with its document + section title. Each chunk "knows" what it is about.
* naive: split purely by character count. Chunks can lose their heading, e.g. a row of the
  savings-rate table ends up separated from the words "High-Yield Savings".
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import EMBED_MODEL, KB_DIR


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    """Read every Markdown file in the knowledge base into a LangChain Document."""
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
    if not docs:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    header_aware: bool = True,
) -> list[Document]:
    """Split documents into retrievable chunks. Each chunk keeps its `source` filename."""
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not header_aware:
        return size_splitter.split_documents(docs)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
    )
    chunks: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            breadcrumb = " > ".join(v for k, v in section.metadata.items() if k in ("title", "section"))
            for piece in size_splitter.split_text(section.page_content):
                chunks.append(
                    Document(
                        # The breadcrumb travels *inside* the text so both the retriever and the
                        # LLM see which product/policy the chunk belongs to.
                        page_content=f"[{breadcrumb}]\n{piece}",
                        metadata={"source": doc.metadata["source"], "section": breadcrumb},
                    )
                )
    return chunks


@lru_cache(maxsize=8)
def build_vector_store(
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    header_aware: bool = True,
    embed_model: str = EMBED_MODEL,
) -> InMemoryVectorStore:
    """Embed the chunks into an in-memory vector store.

    Cached per configuration, so comparing several RAG variants only embeds each variant once.
    In production you would swap InMemoryVectorStore for pgvector, OpenSearch, Pinecone, etc.;
    the evaluation code does not change.
    """
    chunks = split_documents(load_documents(), chunk_size, chunk_overlap, header_aware)
    store = InMemoryVectorStore(OpenAIEmbeddings(model=embed_model))
    store.add_documents(chunks)
    return store
