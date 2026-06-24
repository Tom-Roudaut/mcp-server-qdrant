import openai
from mcp_server_qdrant.embeddings.base import EmbeddingProvider

VECTOR_SIZES = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

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

    def get_vector_name(self) -> str:
        return self.model_name

    def get_vector_size(self) -> int:
        return VECTOR_SIZES.get(self.model_name, 1536)
