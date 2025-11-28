FROM nvidia/cuda:12.2-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.cargo/bin/uv /usr/local/bin/uv
WORKDIR /workspace
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-workspace
COPY . .
RUN uv sync --frozen
ENTRYPOINT ["uv", "run", "train.py"]
