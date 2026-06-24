FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache-dir -e ".[fastembed]"
RUN uv pip install --system --no-cache-dir openai

EXPOSE 8000

ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="startups_global"
ENV OPENAI_API_KEY=""

CMD uvx mcp-server-qdrant --transport streamable-http
