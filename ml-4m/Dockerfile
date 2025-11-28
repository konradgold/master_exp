FROM nvcr.io/nvidia/cuda:12.2-runtime-ubuntu22.04 
RUN apt-get update && apt-get install -y python3-pip curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*  # Clean up
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /workspace
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache uv sync --frozen --no-install-workspace
COPY . .
RUN uv sync --frozen
ENTRYPOINT ["uv", "run", "train.py"]