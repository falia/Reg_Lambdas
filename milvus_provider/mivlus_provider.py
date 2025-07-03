from langchain_milvus import Milvus  # Use the dedicated package, not langchain_community
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class MilvusManager:
    def __init__(self, connection_args: Dict, collection_name: str, host: str, port: str):
        self.connection_args = connection_args
        self.collection_name = collection_name
        self.host = host
        self.port = port

        self.uri = f"tcp://{host}:{int(port)}"
        self.connection_args = {"uri": self.uri}

        self.vector_store = None
        self.collection = None

        # Establish connection to Milvus
        self._connect()

    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                uri=self.uri
            )

            # Verify connection
            addr = connections.get_connection_addr("default")
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            logger.info(f"Milvus connection info: {addr}")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise Exception(f"Milvus connection failed: {e}")

    def create_flattened_schema(self, vector_dim=768):
        """
        Create Milvus collection schema optimized for flattened metadata search
        """
        fields = [
            # Primary key
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),

            # Document content
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),

            # Direct searchable metadata fields
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="document_number", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="super_category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subtitle", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="crawl_timestamp", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="publication_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="update_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="file_size", dtype=DataType.INT64),

            # NEW: Add page_number from unstructured elements
            FieldSchema(name="page_number", dtype=DataType.INT64),

            # Flattened searchable arrays (pipe-separated strings)
            FieldSchema(name="entities_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="keywords_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="themes_text", dtype=DataType.VARCHAR, max_length=2000),

            # Complex metadata as JSON
            FieldSchema(name="complex_metadata", dtype=DataType.JSON),

            # Vector embedding
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Document collection with flattened searchable metadata",
            enable_dynamic_field=True  # Allow additional fields if needed
        )

        return schema

    def create_collection_with_schema(self, embedding_provider, vector_dim=768):
        """Create collection with custom flattened schema and indexes"""
        try:
            # Create schema
            schema = self.create_flattened_schema(vector_dim)

            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )

            logger.info(f"Created collection with flattened schema: {self.collection_name}")

            # Create indexes for better search performance
            self._create_indexes()

            # Load collection
            self.collection.load()

            # Also create langchain vector store for easy integration
            self.vector_store = Milvus(
                collection_name=self.collection_name,
                embedding_function=embedding_provider,
                connection_args=self.connection_args,
                auto_id=True,
            )

            logger.info(f"Successfully created optimized vector store for collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create collection with schema: {e}")
            raise Exception(f"Collection creation failed: {e}")

    def _create_indexes(self):
        """Create indexes on frequently searched fields"""
        try:
            # Index parameters for string fields
            string_index_params = {
                "index_type": "TRIE",
                "params": {}
            }

            # Create indexes on important text fields
            searchable_fields = [
                "title", "document_type", "lang", "super_category",
                "entities_text", "keywords_text", "themes_text", "doc_id"
            ]

            for field in searchable_fields:
                try:
                    self.collection.create_index(
                        field_name=field,
                        index_params=string_index_params
                    )
                    logger.info(f"Created index on field: {field}")
                except Exception as e:
                    logger.warning(f"Could not create index on {field}: {e}")

            # Vector index
            vector_index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }

            self.collection.create_index(
                field_name="vector",
                index_params=vector_index_params
            )

            logger.info("Created vector index")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")

    def create_collection(self, embedding_provider):
        """Create Milvus vector store using langchain_milvus (fallback method)"""
        try:
            # Use langchain_milvus which handles schema creation automatically
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
            # Ensure metadata is properly flattened
            processed_metadatas = []
            for metadata in metadatas or []:
                # If metadata isn't already flattened, log a warning
                if 'complex_metadata' not in metadata:
                    logger.warning("Metadata not flattened - consider using flatten_metadata_for_search()")
                processed_metadatas.append(metadata)

            return self.vector_store.add_texts(texts, metadatas=processed_metadatas)

        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise Exception(f"Failed to add texts to Milvus: {e}")

    def search_by_metadata(self, **kwargs) -> List[Dict]:
        """
        Search by metadata fields using the flattened schema

        Supported search parameters:
        - document_type: exact match
        - lang: exact match
        - super_category: exact match
        - entity: partial match in entities_text
        - keyword: partial match in keywords_text
        - theme: partial match in themes_text
        - title_contains: partial match in title
        - limit: number of results (default 100)
        """
        if not self.collection:
            raise Exception("Collection not initialized with custom schema.")

        conditions = []

        # Direct field searches
        if 'document_type' in kwargs:
            conditions.append(f'document_type == "{kwargs["document_type"]}"')

        if 'lang' in kwargs:
            conditions.append(f'lang == "{kwargs["lang"]}"')

        if 'super_category' in kwargs:
            conditions.append(f'super_category == "{kwargs["super_category"]}"')

        # Text-based searches on flattened arrays
        if 'entity' in kwargs:
            conditions.append(f'entities_text like "%{kwargs["entity"]}%"')

        if 'keyword' in kwargs:
            conditions.append(f'keywords_text like "%{kwargs["keyword"]}%"')

        if 'theme' in kwargs:
            conditions.append(f'themes_text like "%{kwargs["theme"]}%"')

        if 'title_contains' in kwargs:
            conditions.append(f'title like "%{kwargs["title_contains"]}%"')

        if 'doc_id' in kwargs:
            conditions.append(f'doc_id == "{kwargs["doc_id"]}"')

        # Combine conditions
        search_expr = ' and '.join(conditions) if conditions else ""

        try:
            results = self.collection.query(
                expr=search_expr,
                output_fields=["*"],
                limit=kwargs.get('limit', 100)
            )

            return results

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            raise Exception(f"Metadata search failed: {e}")

    def get_full_metadata(self, search_results: List[Dict]) -> List[Dict]:
        """
        Extract and parse the full metadata from search results
        """
        results_with_full_metadata = []

        for result in search_results:
            try:
                # Parse the complex metadata back to original structure
                if 'complex_metadata' in result:
                    complex_meta = json.loads(result['complex_metadata'])

                    # Combine with flattened fields
                    full_metadata = {
                        'doc_id': result.get('doc_id', ''),
                        'title': result.get('title', ''),
                        'document_type': result.get('document_type', ''),
                        'lang': result.get('lang', ''),
                        'entities': complex_meta.get('entities', []),  # Original array
                        'keywords': complex_meta.get('keywords', []),  # Original array
                        'themes': complex_meta.get('themes', []),  # Original array
                        'top_related': complex_meta.get('top_related', []),
                        'content_hash': complex_meta.get('content_hash', ''),
                        'url': result.get('url', ''),
                        'crawl_timestamp': result.get('crawl_timestamp', ''),
                        # Add other fields as needed
                    }

                    result['full_metadata'] = full_metadata

                results_with_full_metadata.append(result)

            except Exception as e:
                logger.error(f"Error parsing metadata for result: {e}")
                result['full_metadata'] = {}
                results_with_full_metadata.append(result)

        return results_with_full_metadata

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

    def hybrid_search(self, query: str, metadata_filters: Dict = None, k: int = 5) -> List[Dict]:
        """
        Perform hybrid search: vector similarity + metadata filtering
        """
        try:
            # First filter by metadata if provided
            if metadata_filters:
                filtered_results = self.search_by_metadata(**metadata_filters, limit=1000)
                if not filtered_results:
                    return []

                # Extract doc_ids from filtered results
                doc_ids = [result['doc_id'] for result in filtered_results]

                # Now do similarity search within these filtered results
                # Note: This is a simplified approach - in production you might want
                # to use Milvus's built-in hybrid search capabilities
                all_results = self.similarity_search_with_score(query, k=k * 2)

                # Filter similarity results to only include docs that match metadata criteria
                hybrid_results = [
                                     result for result in all_results
                                     if result['metadata'].get('doc_id') in doc_ids
                                 ][:k]

                return hybrid_results
            else:
                # Just do similarity search
                return self.similarity_search_with_score(query, k=k)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise Exception(f"Hybrid search failed: {e}")

    def drop_collection(self):
        """Drop the collection"""
        try:
            if self.collection:
                self.collection.drop()
                logger.info(f"Dropped collection: {self.collection_name}")
            elif self.vector_store and hasattr(self.vector_store, 'col'):
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


# Example usage for agents
def example_agent_usage():
    """
    Example of how an agent would use the enhanced MilvusManager
    """

    # Initialize (this would be done in your embedding service)
    milvus_manager = MilvusManager(
        connection_args={"uri": "tcp://34.241.177.15:19530"},
        collection_name="cssf_documents_final",
        host="34.241.177.15",
        port="19530"
    )

    # Example searches an agent might perform:

    # 1. Find all Communiqués about Credit institutions
    results = milvus_manager.search_by_metadata(
        document_type="Communiqué",
        entity="Credit institutions"
    )

    # 2. Find AML/CFT related documents in English
    results = milvus_manager.search_by_metadata(
        keyword="AML/CFT",
        lang="en"
    )

    # 3. Hybrid search: semantic similarity + metadata filtering
    results = milvus_manager.hybrid_search(
        query="banking regulations compliance",
        metadata_filters={
            "document_type": "Communiqué",
            "theme": "Financial crime"
        },
        k=10
    )

    # 4. Get full metadata