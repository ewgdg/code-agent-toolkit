FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv (pin to specific version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set UV link mode to copy for containers and enable bytecode compilation
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock LICENSE README.md ./

# Install dependencies only (not the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/
COPY config/ config/

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Expose port
EXPOSE 8787

# Run the application
CMD ["uv", "run", "claude-router", "--config", "config/router.yaml"]