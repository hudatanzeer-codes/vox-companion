import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


class MemoryManager:
    """
    Handles long-term vector memory retrieval and storage for Vox-Companion 
    using Qdrant vector database and FastEmbed ONNX models.
    """
    def __init__(self, collection_name: str = "vox_memories"):
        self.collection_name = collection_name
        
        logger.info("⚡ Loading FastEmbed ONNX model...")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        if QDRANT_URL == ":memory:":
            logger.info("🧠 Initializing Qdrant in local in-memory mode...")
            self.client = QdrantClient(":memory:")
        else:
            logger.info(f"🧠 Connecting to Qdrant Cluster at {QDRANT_URL}...")
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't already exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info(f"✨ Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    def _get_embedding(self, text: str) -> list[float]:
        """Generates 384-dimensional semantic embedding vectors."""
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()

    def store_memory(self, user_id: str, memory_text: str, metadata: dict = None):
        """Stores a conversational memory into Qdrant."""
        try:
            vector = self._get_embedding(memory_text)
            point_id = int(hashlib.md5(f"{user_id}_{memory_text}".encode()).hexdigest()[:8], 16)
            
            payload = {
                "user_id": user_id,
                "text": memory_text,
                **(metadata or {})
            }

            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)]
            )
            logger.info(f"💾 Stored Memory for [{user_id}]: '{memory_text}'")
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    def retrieve_memories(self, user_id: str, query: str, limit: int = 3, score_threshold: float = 0.52) -> list[str]:
        """Retrieves top relevant memories for the user query above the similarity threshold."""
        try:
            query_vector = self._get_embedding(query)
            
            user_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=user_filter,
                limit=limit,
                score_threshold=score_threshold
            )

            memories = [res.payload["text"] for res in results]
            
            logger.info(f"🔍 Retrieved {len(memories)} relevant memories for query: '{query}'")
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []


if __name__ == "__main__":
    mem = MemoryManager()
    mem.store_memory("user_01", "User prefers vegetarian recipes and simple dishes.")
    mem.store_memory("user_01", "User is studying engineering.")
    
    results = mem.retrieve_memories("user_01", "What do I like to eat?")
    print("\nMemory Retrieval Test Result:")
    print(results)