from langchain_milvus import Milvus  # Use the dedicated package, not langchain_community
from pymilvus import connections
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MilvusManager:
    def __init__(self, connection_args: Dict, collection_name: str, host: str, port: str):
        self.connection_args = connection_args
        self.collection_name = collection_name
        self.host = host
        self.port = int(port)

        self.uri = f"tcp://{host}:{int(port)}"
        self.connection_args = {"uri": self.uri}

        self.vector_store = None

        #self._connect()

    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )

            # Verify connection
            addr = connections.get_connection_addr("default")
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            logger.info(f"Milvus connection info: {addr}")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise Exception(f"Milvus connection failed: {e}")

    def create_collection(self, embedding_provider):
        """Create Milvus vector store using langchain_milvus (the reliable way)"""
        try:

            host = self.connection_args.get("host")
            port = self.connection_args.get("port")
            print(f"Creating Milvus collection with host: {host}, port: {port}")

            self.vector_store = Milvus(
                collection_name=self.collection_name,
                embedding_function=embedding_provider,
                connection_args=self.connection_args,
                auto_id=True,
            )

            logger.info(f"Successfully created vector store for collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise Exception(f"Collection creation failed: {e}")

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """Add texts to the vector store"""
        if not self.vector_store:
            raise Exception("Collection not initialized. Call create_collection() first.")

        try:
            # langchain_milvus handles metadata much better - no need for extensive cleaning
            return self.vector_store.add_texts(texts, metadatas=metadatas)

        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise Exception(f"Failed to add texts to Milvus: {e}")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts"""
        if not self.vector_store:
            raise Exception("Collection not initialized. Call create_collection() first.")

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise Exception(f"Similarity search failed: {e}")

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts with scores"""
        if not self.vector_store:
            raise Exception("Collection not initialized. Call create_collection() first.")

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Similarity search with score failed: {e}")
            raise Exception(f"Similarity search with score failed: {e}")

    def drop_collection(self):
        """Drop the collection"""
        try:
            if self.vector_store and hasattr(self.vector_store, 'col'):
                self.vector_store.col.drop()
                logger.info(f"Dropped collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")

    @classmethod
    def disconnect(cls):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")