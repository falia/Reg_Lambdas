from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import json
from more_itertools import chunked

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler



class EmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


class TEIContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        payload = {"inputs": inputs}
        return json.dumps(payload).encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        if isinstance(response_json, list):
            return response_json
        else:
            raise ValueError(f"Unexpected TEI response format: {type(response_json)}")

class SageMakerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, endpoint_name: str = 'embedding-endpoint', region_name: str = 'eu-west-1', use_tei: bool = True, max_batch_size: int = 8):
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.use_tei = use_tei
        self.max_batch_size = max_batch_size

        content_handler = TEIContentHandler() if use_tei else LegacyContentHandler()

        self.embeddings = SagemakerEndpointEmbeddings(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=content_handler,
        )

    def get_embedding(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for batch in chunked(texts, self.max_batch_size):
            all_embeddings.extend(self.embeddings.embed_documents(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)


class EmbeddingService:
    def __init__(self, use_remote: bool = True, milvus_config: Optional[Dict] = None, use_tei: bool = True, **kwargs):
        self.use_remote = use_remote

        self.provider = SageMakerEmbeddingProvider(use_tei=use_tei, **kwargs)
    
        self.milvus = None
        if milvus_config:
            from milvus_provider.mivlus_provider import MilvusManager

            host = milvus_config.get('host', 'localhost')
            port = milvus_config.get('port', '19530')
            collection_name = milvus_config.get('collection_name', 'embeddings')
            connection_args = milvus_config.get('connection_args', {"host": host, "port": int(port)})

            self.milvus = MilvusManager(
                connection_args=connection_args,
                collection_name=collection_name,
                host=host,
                port=port
            )
            self.milvus.create_collection(self.provider)

    def create_embedding(self, text: str) -> List[float]:
        return self.provider.get_embedding(text)

    def add_text_to_store(self, text: str, metadata: Dict = None) -> Dict:
        if not self.milvus:
            raise Exception("Milvus not configured")

        metadata = metadata or {}
        ids = self.milvus.add_texts([text], [metadata])

        return {
            "text": text,
            "milvus_ids": ids,
            "saved_to_milvus": True,
            "count": 1
        }

    def add_texts_to_store(self, texts: List[str], metadatas: List[Dict] = None) -> Dict:
        if not self.milvus:
            raise Exception("Milvus not configured")

        if not texts:
            return {
                "texts": [],
                "milvus_ids": [],
                "saved_to_milvus": False,
                "count": 0
            }

        ids = self.milvus.add_texts(texts, metadatas)

        return {
            "texts": texts,
            "milvus_ids": ids,
            "saved_to_milvus": True,
            "count": len(texts)
        }

    def search_similar_texts(self, query_text: str, top_k: int = 5, with_scores: bool = False) -> List[Dict]:
        if not self.milvus:
            raise Exception("Milvus not configured")

        if with_scores:
            return self.milvus.similarity_search_with_score(query_text, top_k)
        else:
            return self.milvus.similarity_search(query_text, top_k)

    def switch_provider(self, use_remote: bool, use_tei: bool = True, **kwargs):
        self.use_remote = use_remote
        if use_remote:
            self.provider = SageMakerEmbeddingProvider(use_tei=use_tei, **kwargs)
        else:
            self.provider = LocalEmbeddingProvider(**kwargs)

        if self.milvus:
            self.milvus.create_collection(self.provider)

    def setup_milvus(self, host: str = "localhost", port: str = "19530",
                     connection_args: Dict = None, collection_name: str = "embeddings"):
        from milvus_provider.mivlus_provider import MilvusManager

        connection_args = connection_args or {"host": host, "port": port}

        self.milvus = MilvusManager(
            connection_args=connection_args,
            collection_name=collection_name,
            host=host,
            port=port
        )
        self.milvus.create_collection(self.provider)
