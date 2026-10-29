"""Stage 3 - Context-engineered chunking + metadata-filtered retrieval with LangChain.

Key ideas
* Tables are chunked as atomic units (split only on row boundaries, header repeated), prose via a
  recursive splitter - so tables never get scrambled into word soup.
* Every chunk carries metadata {drug, phase, therapeutic_area, sponsor, access_level, doc_id, page, kind}
  and a one-line context header is prepended to the embedded text ("context engineering").
* `SecureRetriever` ENFORCES filters: the caller's role determines the allowed access levels and that
  constraint is AND-ed with any requested phase/drug filter. The query text can't widen access.

Embeddings: offline hashing embeddings by default (no API key); set OPENAI_API_KEY and
EMBEDDINGS=openai to use OpenAI embeddings instead.
"""
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
ELEMENTS = ROOT / "data" / "processed" / "elements.parquet"

ROLE_ACCESS = {                       # role -> access levels it may read
    "public_user": {"public"},
    "employee": {"public", "internal"},
    "admin": {"public", "internal", "restricted"},
}
META_KEYS = ["drug", "phase", "therapeutic_area", "sponsor", "access_level"]


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words(+bigram) hashing embeddings - dependency-free, good enough for a demo."""

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        toks = re.findall(r"[a-z0-9\-\.]+", text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        v = [1 + math.log(x) if x > 0 else 0.0 for x in v]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embeddings() -> Embeddings:
    if os.getenv("EMBEDDINGS") == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HashingEmbeddings()


def _split_table(md: str, max_chars: int) -> List[str]:
    lines = md.split("\n")
    header, rows = lines[:2], lines[2:]
    chunks, cur = [], []
    for r in rows:
        if cur and len("\n".join(header + cur + [r])) > max_chars:
            chunks.append("\n".join(header + cur))
            cur = []
        cur.append(r)
    if cur:
        chunks.append("\n".join(header + cur))
    return chunks or [md]


def build_documents(df: pd.DataFrame, chunk_chars: int = 1200, overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    docs: List[Document] = []
    for r in df.itertuples():
        meta = {k: getattr(r, k) for k in META_KEYS}
        meta.update(doc_id=r.doc_id, page=int(r.page), kind=r.kind, title=r.title)
        pieces = _split_table(r.text, chunk_chars * 2) if r.kind == "table" else splitter.split_text(r.text)
        for i, p in enumerate(pieces):
            header = (f"[{r.drug} | {r.phase} | {r.therapeutic_area} | Sponsor: {r.sponsor} | "
                      f"{r.doc_id} p.{r.page} | {r.kind}]")
            docs.append(Document(page_content=f"{header}\n{p}", metadata={**meta, "chunk": i}))
    return docs


def _phase_labels(phase: str) -> set:
    """'Phase I/II' -> {'phase i/ii', 'phase i', 'phase ii'} so a Phase II query matches Phase I/II trials."""
    p = phase.lower()
    parts = p.replace("phase", "").strip().split("/")
    return {p} | {f"phase {x.strip()}" for x in parts if x.strip()}


class SecureRetriever:
    """Retriever that always applies a role-derived access filter plus optional metadata filters."""

    def __init__(self, store: InMemoryVectorStore, role: str, k: int = 4):
        if role not in ROLE_ACCESS:
            raise PermissionError(f"unknown role {role!r}")
        self.store, self.role, self.k = store, role, k

    def _predicate(self, phase: Optional[str], drug: Optional[str], area: Optional[str],
                   kind: Optional[str]) -> Callable[[Document], bool]:
        allowed = ROLE_ACCESS[self.role]

        def ok(d: Document) -> bool:
            m = d.metadata
            if m["access_level"] not in allowed:                     # mandatory authorization
                return False
            if phase and phase.lower() not in _phase_labels(m["phase"]):
                return False
            if drug and drug.lower() not in str(m["drug"]).lower():
                return False
            if area and area.lower() not in str(m["therapeutic_area"]).lower():
                return False
            return not (kind and m["kind"] != kind)
        return ok

    def retrieve(self, query: str, *, phase=None, drug=None, area=None, kind=None) -> List[Document]:
        return self.store.similarity_search(query, k=self.k, filter=self._predicate(phase, drug, area, kind))


def build_store(df: Optional[pd.DataFrame] = None) -> InMemoryVectorStore:
    df = pd.read_parquet(ELEMENTS) if df is None else df
    store = InMemoryVectorStore(get_embeddings())
    store.add_documents(build_documents(df))
    return store
