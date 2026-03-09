# Semantic Cache Middleware

> Stop wasting money on repetitive LLM calls.

A lightweight Rust middleware that intercepts LLM API requests, embeds them, and checks a local vector store. If a semantically similar question was asked before, it returns the cached answer instantly — saving tokens, latency, and cost.

## Core Concepts

| Concept | Role |
|---|---|
| **Vector Math** | Embed prompts into high-dimensional vectors |
| **Cosine Similarity** | Measure how alike two prompts are |
| **Concurrency (Tokio)** | Handle many requests without blocking |
| **Hashing** | Fast exact-match pre-filter before vector search |

The similarity between two prompt vectors **A** and **B** is computed as:

$$
\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}
$$

## Architecture

```
Client
  │
  ▼
┌──────────────────────────┐
│   Axum REST API          │  POST /generate { "prompt": "..." }
│   (tower-http limits)    │
├──────────────────────────┤
│   Semantic Cache         │  Embed (rust-bert) → search HNSW (hnsw_rs)
│   (moka + hnsw_rs)       │  → Hit? return cached response
├──────────────────────────┤
│   LLM Service            │  On miss → forward to LLM, update cache
└──────────────────────────┘
```

### Project Structure

```
src/
├── main.rs          # Bootstrap: config → service → routes → serve
├── config.rs        # AppConfig (env vars with defaults)
├── error.rs         # AppError enum → Axum IntoResponse
├── models.rs        # Request / response DTOs
├── handlers.rs      # Route handlers (thin, delegate to services)
├── routes.rs        # Router construction + state wiring
└── services/
    ├── mod.rs       # Re-exports
    └── ollama.rs    # Ollama HTTP client wrapper
```

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (edition 2024)
- [Ollama](https://ollama.com/) running locally (default: `http://localhost:11434`)

### Run

```bash
# Clone the repo
git clone https://github.com/LyesHADJAR/semantic-cache-middleware.git
cd semantic-cache-middleware

# Start Ollama (in another terminal)
ollama serve

# Run the server
cargo run
```

The server starts at **http://127.0.0.1:3000** by default.

### Configuration

All settings are read from environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `LISTEN_ADDR` | `127.0.0.1:3000` | Address the server binds to |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2` | Model used for generation |

```bash
OLLAMA_MODEL=mistral LISTEN_ADDR=0.0.0.0:8080 cargo run
```

## API

### `GET /`

Health check.

```bash
curl http://localhost:3000/
# Hello, World!
```

### `POST /generate`

Send a prompt to the LLM.

```bash
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain ownership in Rust"}'
```

*(Note: Requires a populated local Ollama `llama3.2` model to avoid upstream errors, and requests over 2 MB or 100,000 characters will be actively safely rejected)*

## Roadmap

- [x] REST API accepting prompt strings (Axum)
- [x] Compute embeddings using `rust-bert` natively in a background worker
- [x] In-memory semantic cache with `moka`
- [x] O(N) Cosine similarity search for cache hits via `hnsw_rs`
- [x] Configurable similarity threshold
- [x] Cache TTL / graceful background eviction policies
- [x] Security policies (Request length limits, Timeouts, empty prompt guards)

## Tech Stack

- **[Axum](https://github.com/tokio-rs/axum)** — async web framework
- **[Tokio](https://tokio.rs/)** — async runtime
- **[reqwest](https://github.com/seanmonstar/reqwest)** — HTTP API client
- **[Ollama](https://ollama.com/)** — local LLM inference
- **[rust-bert](https://github.com/guillaume-be/rust-bert)** — native embedding model operations
- **[moka](https://github.com/moka-rs/moka)** — bounded high-performance concurrent cache
- **[hnsw_rs](https://github.com/jean-pierreCorriveau/hnsw_rs)** — ANN similarity search

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
