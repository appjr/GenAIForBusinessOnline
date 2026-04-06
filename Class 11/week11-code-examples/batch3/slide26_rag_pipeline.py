"""
Week 11: Slide 26 — RAG Pipeline for Business Knowledge Q&A

Description:
    Builds a complete Retrieval-Augmented Generation (RAG) pipeline that
    allows an LLM to answer questions grounded in a private document corpus.
    Demonstrates embedding, vector storage, retrieval, and response generation
    with proper citation tracking.

    This is the recommended pattern for "chat with your company's documents"
    use cases — HR policies, product manuals, research reports, etc.

Dependencies:
    - anthropic
    - chromadb
    - python-dotenv

    (Optional for PDF support):
    - pypdf

Usage:
    python slide26_rag_pipeline.py

    For more documents, add .txt or .pdf files to the ./documents/ folder
    before running.
"""

import os
import json
import textwrap
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()


# ─── Configuration ─────────────────────────────────────────────────────────────

CHUNK_SIZE = 500        # Characters per document chunk
CHUNK_OVERLAP = 100     # Overlap between chunks to preserve context
TOP_K = 4              # Number of chunks to retrieve per query
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Local model, no API needed for embeddings
LLM_MODEL = "claude-haiku-4-5-20251001"         # Fast + cheap for RAG responses


# ─── Sample Documents ──────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = {
    "ai_strategy_2026.txt": textwrap.dedent("""
        ACME Corp AI Strategy — 2026

        Executive Summary
        -----------------
        ACME Corp is committed to responsible AI adoption across all business units.
        Our 2026 AI strategy focuses on three pillars: productivity enhancement,
        customer experience improvement, and operational efficiency.

        Approved AI Tools
        -----------------
        The following AI tools are approved for use by ACME employees:
        - Claude (Anthropic) — for writing, analysis, and coding tasks
        - GitHub Copilot — for software development teams
        - Zapier AI — for workflow automation (non-sensitive data only)
        - Google Gemini Workspace — integrated in Google Suite

        Data Classification for AI Use
        ------------------------------
        Class 1 (Public): May be shared with any AI tool
        Class 2 (Internal): May be used with approved tools only
        Class 3 (Confidential): Must use on-premises AI only (no cloud APIs)
        Class 4 (Restricted): AI processing not permitted

        AI Investment Budget
        --------------------
        Total AI budget for 2026: $2.4M
        - Tools and subscriptions: $800K
        - Custom development: $1.2M
        - Training and enablement: $400K

        Success Metrics
        ---------------
        We will measure success by:
        - 20% reduction in time spent on routine analysis tasks
        - 15% improvement in customer response time
        - 25% faster software development cycles
    """).strip(),

    "data_governance_policy.txt": textwrap.dedent("""
        ACME Corp Data Governance Policy — Version 4.2

        Purpose
        -------
        This policy establishes the framework for managing data assets at ACME Corp,
        ensuring data quality, security, and compliance with applicable regulations.

        Data Ownership
        --------------
        Every data asset must have a designated Data Owner who is responsible for:
        - Classifying data according to the Data Classification Framework
        - Approving access requests for their data domain
        - Ensuring data quality and accuracy

        Data Retention Standards
        ------------------------
        Customer data: Retain for 7 years after last transaction
        Employee data: Retain for 5 years after employment ends
        Financial records: Retain for 10 years per regulatory requirement
        Marketing data: Retain for 3 years unless customer consent renewed

        AI and Machine Learning Data
        ----------------------------
        Training datasets must be:
        - Documented in the central data catalog
        - Reviewed for bias before use in production models
        - Retained for the lifetime of the model plus 2 years
        - Deleted within 90 days of model decommissioning

        Breach Notification
        -------------------
        In the event of a data breach:
        1. IT Security team must be notified within 1 hour of discovery
        2. Legal team must assess notification obligations within 4 hours
        3. Affected customers must be notified within 72 hours if required by law
        4. Regulatory authorities must be notified within 72 hours (GDPR, CCPA)
    """).strip(),

    "ml_model_deployment_guidelines.txt": textwrap.dedent("""
        ACME Corp ML Model Deployment Guidelines

        Overview
        --------
        These guidelines apply to all machine learning models deployed to production
        systems at ACME Corp, including LLM-based applications and traditional ML models.

        Pre-Deployment Checklist
        ------------------------
        Before any model is deployed to production, teams must complete:
        [ ] Model card documentation (accuracy, bias testing, limitations)
        [ ] Security review by InfoSec team
        [ ] Privacy impact assessment for models processing personal data
        [ ] Performance benchmarking vs. baseline
        [ ] Rollback plan documented
        [ ] Monitoring dashboards configured

        Model Monitoring Requirements
        -----------------------------
        All production models must be monitored for:
        - Data drift: alert if input distribution shifts > 10% from training data
        - Prediction drift: alert if output distribution shifts unexpectedly
        - Latency: p99 latency must remain under SLA thresholds
        - Error rates: page on-call if error rate exceeds 1%

        LLM-Specific Requirements
        -------------------------
        For Large Language Model deployments:
        - System prompts must be reviewed by Legal and Product teams
        - Outputs must be logged for 30 days for audit purposes
        - Hallucination rate must be measured and documented
        - Human review required for high-stakes outputs (medical, legal, financial)
        - Cost monitoring: alert if daily token spend exceeds $500

        Responsible AI Review Board
        ----------------------------
        Models in the following categories require RARB approval before deployment:
        - Customer-facing AI making automated decisions
        - Models trained on sensitive or protected class data
        - AI systems with potential for significant societal impact
        - Any model replacing a human decision-making process
    """).strip(),
}


# ─── Document Processing ───────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A chunk of a document, ready for embedding and retrieval.

    Attributes:
        text: The actual text content of the chunk.
        source: Filename the chunk came from.
        chunk_id: Sequential chunk number within the document.
        doc_id: Unique identifier for this chunk (hash of content).
    """
    text: str
    source: str
    chunk_id: int
    doc_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.doc_id = hashlib.md5(f"{self.source}:{self.chunk_id}:{self.text[:50]}".encode()).hexdigest()


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks for indexing.

    Args:
        text: Full document text.
        source: Source filename for citation.
        chunk_size: Target characters per chunk.
        overlap: Overlap in characters between adjacent chunks.

    Returns:
        List of DocumentChunk objects.

    Example:
        >>> chunks = chunk_text("Long document...", "policy.txt")
        >>> len(chunks)
        3
    """
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_content = text[start:end]

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = chunk_text_content.rfind(".")
            if last_period > chunk_size // 2:  # Only break at sentence if reasonable
                chunk_text_content = chunk_text_content[:last_period + 1]
                end = start + last_period + 1

        chunks.append(DocumentChunk(
            text=chunk_text_content.strip(),
            source=source,
            chunk_id=chunk_id
        ))

        start = end - overlap
        chunk_id += 1

    return chunks


# ─── Vector Store ─────────────────────────────────────────────────────────────

class DocumentStore:
    """
    A vector store for RAG — handles indexing and retrieval of document chunks.

    Uses ChromaDB (local, embedded) with sentence-transformer embeddings.
    No API keys needed for embeddings — everything runs locally.

    Example:
        >>> store = DocumentStore()
        >>> store.add_document("policy.txt", "Our policy states...")
        >>> results = store.search("What is the retention policy?")
    """

    def __init__(self, collection_name: str = "business_docs"):
        """
        Initialize the vector store.

        Args:
            collection_name: ChromaDB collection to use.
        """
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.client = chromadb.Client()

        # Delete and recreate for clean demo
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self.indexed_sources: list[str] = []

    def add_document(self, filename: str, content: str) -> int:
        """
        Index a document by chunking it and adding to the vector store.

        Args:
            filename: Source document name (used for citations).
            content: Full text content to index.

        Returns:
            Number of chunks indexed.
        """
        chunks = chunk_text(content, filename)

        if not chunks:
            return 0

        self.collection.add(
            ids=[c.doc_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "chunk_id": c.chunk_id} for c in chunks]
        )
        self.indexed_sources.append(filename)
        return len(chunks)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: Natural language question or search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'text', 'source', and 'distance' keys.

        Example:
            >>> results = store.search("What is the data retention policy?")
            >>> for r in results:
            ...     print(r['source'], r['text'][:100])
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        chunks = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                chunks.append({
                    "text": doc,
                    "source": meta["source"],
                    "distance": dist
                })

        return chunks


# ─── RAG Query Engine ─────────────────────────────────────────────────────────

class RAGQueryEngine:
    """
    Combines retrieval and generation to answer questions from documents.

    Args:
        store: A populated DocumentStore.
        model: Claude model to use for generation.
    """

    def __init__(self, store: DocumentStore, model: str = LLM_MODEL):
        self.store = store
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        verbose: bool = False
    ) -> dict[str, str]:
        """
        Answer a question using retrieved document context.

        Args:
            question: Natural language question.
            top_k: Number of context chunks to retrieve.
            verbose: Print retrieved chunks to stdout.

        Returns:
            Dict with 'answer', 'sources', and 'context_used' keys.

        Example:
            >>> engine = RAGQueryEngine(store)
            >>> result = engine.query("What is the data breach notification timeline?")
            >>> print(result['answer'])
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.store.search(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "No relevant documents found for this question.",
                "sources": [],
                "context_used": ""
            }

        if verbose:
            print(f"\n[Retrieval] Found {len(chunks)} relevant chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. [{chunk['source']}] similarity={1-chunk['distance']:.3f}")
                print(f"     {chunk['text'][:100]}...")

        # Step 2: Build context string with source attribution
        context_parts = []
        sources_seen = set()
        for chunk in chunks:
            context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
            sources_seen.add(chunk["source"])

        context = "\n\n---\n\n".join(context_parts)
        sources = sorted(sources_seen)

        # Step 3: Generate answer grounded in context
        prompt = textwrap.dedent(f"""
            You are a helpful business assistant answering questions based on company documents.

            INSTRUCTIONS:
            - Answer ONLY using the provided context below
            - If the answer isn't in the context, say "I don't have information about that in the provided documents"
            - Be specific and cite which document your answer comes from
            - Keep answers concise (2–4 sentences unless more detail is needed)

            CONTEXT:
            {context}

            QUESTION: {question}

            ANSWER:
        """).strip()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.content[0].text.strip(),
            "sources": sources,
            "context_used": context[:500] + "..."
        }


# ─── Main ──────────────────────────────────────────────────────────────────────

def setup_demo_documents() -> None:
    """Create sample document files for the demo."""
    docs_dir = Path("./documents")
    docs_dir.mkdir(exist_ok=True)
    for filename, content in SAMPLE_DOCUMENTS.items():
        (docs_dir / filename).write_text(content, encoding="utf-8")
    print(f"Created {len(SAMPLE_DOCUMENTS)} sample documents in ./documents/")


if __name__ == "__main__":
    print("RAG Pipeline Demo — Week 11")
    print("=" * 60)

    # Step 1: Set up sample documents
    setup_demo_documents()

    # Step 2: Build the vector store
    print("\nIndexing documents...")
    store = DocumentStore()
    total_chunks = 0
    for filename, content in SAMPLE_DOCUMENTS.items():
        n = store.add_document(filename, content)
        total_chunks += n
        print(f"  Indexed: {filename} ({n} chunks)")
    print(f"\nTotal: {total_chunks} chunks across {len(SAMPLE_DOCUMENTS)} documents")

    # Step 3: Build the query engine
    engine = RAGQueryEngine(store)

    # Step 4: Ask business questions
    questions = [
        "What AI tools are approved for ACME employees?",
        "What is the timeline for notifying customers after a data breach?",
        "What monitoring is required for production ML models?",
        "What is the AI budget for 2026?",
        "When does a model need Responsible AI Review Board approval?",
        "What data classification level is required to use cloud AI APIs?",
    ]

    print("\n" + "=" * 60)
    print("Q&A DEMO")
    print("=" * 60)

    for question in questions:
        result = engine.query(question, verbose=False)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"   Sources: {', '.join(result['sources'])}")

    # Step 5: Interactive mode (optional)
    print("\n" + "=" * 60)
    print("Try your own question (or press Enter to skip):")
    user_q = input("> ").strip()
    if user_q:
        result = engine.query(user_q, verbose=True)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
