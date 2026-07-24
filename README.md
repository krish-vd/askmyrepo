# AskMyRepo

Semantic code search over any GitHub repo. Paste a repo URL, ask a question in plain English, get an answer grounded in the actual code — with `file:line` citations, not guesses.

Runs entirely locally: local embeddings, local LLM, no external API calls, no API costs.

## How it works

1. **Clone & chunk** — shallow-clones the repo and walks its files (skipping vendor/binary/oversized paths).
2. **AST-aware chunking** — Python and JS/TS files are parsed with [tree-sitter](https://tree-sitter.github.io/tree-sitter/) and split by real function/class boundaries, so a chunk is never a truncated function. Other files, or files that fail to parse, fall back to a line-window chunker.
3. **Embed** — each chunk is embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) and stored in Postgres via [pgvector](https://github.com/pgvector/pgvector).
4. **Call-graph pass** — a lightweight heuristic links chunks that call each other by symbol name, so a retrieved function's direct callers/callees can be pulled in as extra context.
5. **Retrieve** — a question is embedded and matched against stored chunks by cosine similarity (top-k, similarity-cutoff filtered), then expanded with call-graph neighbors up to a capped context size.
6. **Generate** — the retrieved chunks are passed to a local LLM ([Ollama](https://ollama.com), `qwen2.5-coder:3b`) with instructions to answer only from the given context and cite `file:line` for every claim. Temperature, top-p, and top-k are exposed as tunable parameters, not hidden defaults.

The UI also visualizes the index as a live graph — every node is a real chunk, and asking a question highlights exactly which chunks were retrieved and used.

## Stack

- Django 6 + PostgreSQL 17 + pgvector
- tree-sitter (Python, JS/TS grammars)
- sentence-transformers (`all-MiniLM-L6-v2`)
- Ollama (`qwen2.5-coder:3b`)

## Setup

Requires Python 3.11+, PostgreSQL with the `vector` extension, and [Ollama](https://ollama.com).

```bash
# 1. Postgres + pgvector
createdb askmyrepo
psql askmyrepo -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 2. Local LLM
ollama pull qwen2.5-coder:3b

# 3. Python environment
python3 -m venv venv
source venv/bin/activate
pip install django "psycopg[binary]" pgvector sentence-transformers gitpython \
            tree-sitter tree-sitter-python tree-sitter-javascript requests

# 4. Configure
cp .env.example .env
# edit .env with your DB_NAME/DB_USER/DB_PASSWORD/DB_HOST/DB_PORT

# 5. Migrate + run
python manage.py migrate
python manage.py runserver
```

Open `http://localhost:8000`, paste a small public GitHub repo URL, and ask it a question once ingestion finishes.

## Notes on scope

This is a weekend build, not a production service — a few deliberate simplifications:

- Ingestion runs synchronously in the request (bounded by `MAX_INGEST_FILES`); a production version would hand this off to a task queue (Celery/RQ).
- The call graph is a regex heuristic (does symbol `X` appear as `X(` in another chunk's source), not real static analysis — it will have false positives/negatives.
- No auth, rate limiting, or multi-user support.
- The graph visualization caps at 300 nodes for render performance; retrieval itself always searches the full indexed set, regardless of repo size.
