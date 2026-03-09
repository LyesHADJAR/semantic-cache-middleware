# ─────────────────────────────────────────────────────
# Stage 1: Build
# ─────────────────────────────────────────────────────
FROM rust:1.87-bookworm AS builder

# Install libtorch build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake pkg-config libssl-dev g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency compilation: copy manifests first.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    echo "" > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy the real source and build.
COPY . .
# Touch main.rs so cargo detects the change.
RUN touch src/main.rs src/lib.rs && cargo build --release

# ─────────────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user.
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --create-home appuser

WORKDIR /app

# Copy the compiled binary from builder.
COPY --from=builder /app/target/release/semantic-cache-middleware .

# Copy libtorch shared objects that tch-rs downloaded during build.
COPY --from=builder /app/target/release/build/torch-sys-*/out/libtorch/lib/*.so* /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

# Drop privileges.
USER appuser

EXPOSE 3000

# Health-check against the root endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["sh", "-c", "exec 3<>/dev/tcp/localhost/3000 && echo -e 'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n' >&3 && timeout 2 cat <&3 | head -1 | grep -q '200'"]

ENTRYPOINT ["./semantic-cache-middleware"]
