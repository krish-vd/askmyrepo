"""Query embedding -> top-k pgvector retrieval -> call-graph expansion ->
prompt construction -> local LLM generation via Ollama.
"""
import requests
from pgvector.django import CosineDistance

from .embedding import embed_query
from .models import CodeChunk

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5-coder:3b"

MAX_CONTEXT_CHUNKS = 8
SIMILARITY_CUTOFF = 0.15  # cosine similarity (1 - distance); below this a hit is noise, not signal

SYSTEM_PROMPT = (
    "You are a code assistant answering questions about a specific repository. "
    "Answer ONLY using the code context provided below — do not invent functions, "
    "classes, or behavior that is not shown. For every claim you make, cite the "
    "source as `file_path:start_line`. If the provided context does not contain "
    "enough information to answer the question, say so explicitly instead of "
    "guessing."
)


def retrieve_chunks(repo, question: str, top_k: int = 4):
    """Embed the question and return the top_k most similar chunks in repo,
    each annotated with a cosine similarity score, filtered by SIMILARITY_CUTOFF."""
    query_vec = embed_query(question)
    qs = (
        CodeChunk.objects.filter(repo=repo, embedding__isnull=False)
        .annotate(distance=CosineDistance("embedding", query_vec))
        .order_by("distance")[:top_k]
    )
    hits = []
    for chunk in qs:
        similarity = 1 - chunk.distance
        if similarity < SIMILARITY_CUTOFF:
            continue
        hits.append({"chunk": chunk, "similarity": similarity, "source": "direct"})
    return hits


def expand_with_call_graph(hits, max_total=MAX_CONTEXT_CHUNKS):
    """Add each direct hit's callers/callees to the context set, capped at
    max_total chunks overall. Expanded chunks carry no similarity score since
    they weren't ranked — they're included for call-chain context."""
    seen_ids = {h["chunk"].id for h in hits}
    expanded = list(hits)

    for hit in hits:
        if len(expanded) >= max_total:
            break
        related = list(hit["chunk"].calls.all()) + list(hit["chunk"].called_by.all())
        for rel in related:
            if len(expanded) >= max_total:
                break
            if rel.id in seen_ids:
                continue
            seen_ids.add(rel.id)
            expanded.append({"chunk": rel, "similarity": None, "source": "call_graph"})

    return expanded


def build_prompt(question: str, context_chunks):
    blocks = []
    for item in context_chunks:
        c = item["chunk"]
        label = c.symbol_name or f"({c.symbol_type})"
        blocks.append(
            f"--- {c.file_path}:{c.start_line}-{c.end_line} [{c.symbol_type} {label}] ---\n"
            f"{c.source_code}"
        )
    context_text = "\n\n".join(blocks)
    user_prompt = (
        f"Code context:\n\n{context_text}\n\n"
        f"Question: {question}"
    )
    return user_prompt


def generate_answer(
    question: str,
    context_chunks,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
):
    """Call Ollama's chat endpoint with the given context and sampling params.
    temperature/top_p/top_k are surfaced as explicit args (not buried in a
    config object) since tuning/showing them is a deliberate project goal."""
    user_prompt = build_prompt(question, context_chunks)
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def answer_question(
    repo,
    question: str,
    top_k_retrieval: int = 4,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k_sampling: int = 40,
):
    """Full retrieval + generation pipeline. Returns a dict with the answer,
    the retrieval trace (for the UI's retrieval-log panel), and the
    generation params used."""
    direct_hits = retrieve_chunks(repo, question, top_k=top_k_retrieval)
    context_chunks = expand_with_call_graph(direct_hits)
    answer = generate_answer(
        question,
        context_chunks,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_sampling,
    )

    trace = []
    for item in context_chunks:
        c = item["chunk"]
        trace.append({
            "id": c.id,
            "file_path": c.file_path,
            "symbol_name": c.symbol_name,
            "symbol_type": c.symbol_type,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "similarity": round(item["similarity"], 4) if item["similarity"] is not None else None,
            "source": item["source"],
        })

    return {
        "answer": answer,
        "retrieved_chunks": trace,
        "params": {
            "top_k_retrieval": top_k_retrieval,
            "temperature": temperature,
            "top_p": top_p,
            "top_k_sampling": top_k_sampling,
        },
    }
