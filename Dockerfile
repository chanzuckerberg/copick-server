FROM ghcr.io/astral-sh/uv:debian-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential

# required for pytree to run
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
# Add Rust to PATH (for current shell)
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /app

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

RUN uv sync --frozen

CMD ["./start_webapp.sh"]