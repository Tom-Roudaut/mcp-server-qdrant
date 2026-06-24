import openai
from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.OpenAI()

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=documents,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[query],
            model=self.model_name
        )
        return response.data[0].embedding
