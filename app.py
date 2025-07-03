# Enhanced RAG system with simplified circular comparison
import json
import re
from typing import List, Dict, Any, Optional
import os
from langchain_aws import BedrockLLM
from langchain_core.documents import Document
from embedding_provider.embedding_provider import EmbeddingService

# Global variables for service reuse
llm = None
embedding_service = None


class CircularComparator:
    """Simplified circular comparison using only metadata fields"""

    def __init__(self, embedding_service, llm):
        self.embedding_service = embedding_service
        self.llm = llm

    def find_circular_by_number(self, circular_number: str) -> List[Dict]:
        """Find all chunks for a specific circular using direct metadata search"""
        try:
            normalized_number = self._normalize_circular_number(circular_number)

            try:
                from pymilvus import Collection
                collection = Collection(self.embedding_service.milvus.collection_name)
                query_expr = f'document_type == "CSSF circular" and document_number == "{normalized_number}"'

                results = collection.query(
                    expr=query_expr,
                    output_fields=["*"],
                    limit=10000
                )

                if results:
                    formatted_results = []
                    for result in results:
                        doc = {
                            'content': result.get('text', ''),
                            'text': result.get('text', ''),
                            'metadata': {
                                'doc_id': result.get('doc_id', ''),
                                'title': result.get('title', ''),
                                'document_type': result.get('document_type', ''),
                                'document_number': result.get('document_number', ''),
                                'lang': result.get('lang', ''),
                                'url': result.get('url', ''),
                                'publication_date': result.get('publication_date', ''),
                            }
                        }
                        formatted_results.append(doc)
                    return formatted_results
                else:
                    return []

            except Exception:
                # Fallback to semantic search + filtering
                results = self.embedding_service.search_similar_texts(
                    query_text=f"CSSF circular {normalized_number}",
                    top_k=200,
                    with_scores=True
                )

                filtered_results = []
                for doc_data in results:
                    if isinstance(doc_data, tuple):
                        doc_content, score = doc_data
                    else:
                        doc_content = doc_data

                    metadata = doc_content.get('metadata', {})
                    doc_type = metadata.get('document_type', '')
                    doc_number = metadata.get('document_number', '')
                    title = metadata.get('title', '')

                    if (doc_type == "CSSF circular" and
                            (doc_number == normalized_number or normalized_number in title)):
                        filtered_results.append(doc_content)

                return filtered_results

        except Exception:
            return []

    def _normalize_circular_number(self, circular_number: str) -> str:
        """Normalize circular number to match metadata format"""
        number = circular_number.strip()

        if number.startswith("CSSF"):
            return number

        if "/" in number:
            number = number.replace("/", "-")

        if not number.startswith("CSSF"):
            number = f"CSSF {number}"

        return number

    def extract_circular_numbers_from_query(self, query: str) -> List[str]:
        """Extract circular numbers from query using regex patterns"""
        patterns = [
            r'CSSF\s+(\d{1,2}[-/]\d{3})',
            r'\b(\d{1,2}[-/]\d{3})\b',
            r'circular\s+(\d{1,2}[-/]\d{3})',
        ]

        found_numbers = []
        query_lower = query.lower()

        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if not match.startswith('CSSF'):
                    normalized = f"CSSF {match}".replace('/', '-')
                else:
                    normalized = match.replace('/', '-')

                if normalized not in found_numbers:
                    found_numbers.append(normalized)

        return found_numbers

    def compare_two_circulars(self, circular1_number: str, circular2_number: str, query: str) -> str:
        """Compare two circulars by their document numbers"""
        try:
            circular1_chunks = self.find_circular_by_number(circular1_number)
            circular2_chunks = self.find_circular_by_number(circular2_number)

            if not circular1_chunks:
                return f"Could not find circular {circular1_number} in the database."

            if not circular2_chunks:
                return f"Could not find circular {circular2_number} in the database."

            content1 = self._extract_content_from_chunks(circular1_chunks)
            content2 = self._extract_content_from_chunks(circular2_chunks)

            title1 = self._get_title_from_chunks(circular1_chunks) or circular1_number
            title2 = self._get_title_from_chunks(circular2_chunks) or circular2_number

            comparison_result = self._generate_comparison(content1, content2, title1, title2)

            return comparison_result

        except Exception as e:
            return f"Error occurred while comparing circulars: {str(e)}"

    def _extract_content_from_chunks(self, chunks: List[Dict]) -> str:
        """Extract and combine content from all chunks"""
        content_parts = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                content = (chunk.get('content') or
                           chunk.get('text') or
                           chunk.get('page_content') or
                           '')
                if content:
                    content_parts.append(content)

        return '\n\n'.join(content_parts)

    def _get_title_from_chunks(self, chunks: List[Dict]) -> str:
        """Extract title from chunk metadata"""
        for chunk in chunks:
            if isinstance(chunk, dict):
                metadata = chunk.get('metadata', {})
                title = metadata.get('title')
                if title:
                    return title
        return None

    def _generate_comparison(self, content1: str, content2: str, title1: str, title2: str) -> str:
        """Generate comparison using LLM"""
        try:
            # Test basic connectivity first
            basic_response = self.llm.invoke("Say 'Test successful'")
            basic_content = extract_content_from_response(basic_response)

            if not basic_content.strip():
                return self._generate_fallback_comparison(content1, content2, title1, title2)

            # Try structured comparison
            content_prompt = f"""
            Compare these CSSF circulars:

            Circular A: {title1}
            Content sample: {content1[:500]}

            Circular B: {title2}  
            Content sample: {content2[:500]}

            Provide a structured comparison:

            ## Key Differences:
            ## Status:
            ## Impact for institutions:
            """

            content_response = self.llm.invoke(content_prompt)
            content_result = extract_content_from_response(content_response)

            if content_result.strip() and len(content_result) > 50:
                return content_result
            else:
                return self._generate_fallback_comparison(content1, content2, title1, title2)

        except Exception:
            return self._generate_fallback_comparison(content1, content2, title1, title2)

    def _generate_fallback_comparison(self, content1: str, content2: str, title1: str, title2: str) -> str:
        """Generate a fallback comparison when LLM fails"""
        content1_words = len(content1.split())
        content2_words = len(content2.split())

        return f"""
## Circular Comparison: {title1} vs {title2}

### Document Analysis
- **{title1}**: {content1_words} words
- **{title2}**: {content2_words} words

### Basic Observations
- Circular 1 appears to be {"longer" if content1_words > content2_words else "shorter"} than Circular 2
- Both circulars contain regulatory language and compliance requirements

### Content Preview
**From {title1}:**
{content1[:300]}...

**From {title2}:**
{content2[:300]}...

### Recommendation
For a detailed regulatory analysis, please:
1. Review the full circular documents directly
2. Consult with legal or compliance professionals
3. Check the CSSF website for the most current versions
"""

    def process_comparison_query(self, query: str) -> Dict[str, Any]:
        """Main method to process circular comparison queries"""
        try:
            circular_numbers = self.extract_circular_numbers_from_query(query)

            if len(circular_numbers) < 2:
                return {
                    'answer': "I need at least two circular numbers to compare. Please specify both circulars in your query (e.g., 'compare CSSF 16-635 and CSSF 12-539').",
                    'sources': [],
                    'workflow': 'comparison_insufficient_numbers',
                    'found_numbers': circular_numbers
                }

            circular1_number = circular_numbers[0]
            circular2_number = circular_numbers[1]

            comparison_result = self.compare_two_circulars(circular1_number, circular2_number, query)

            return {
                'answer': comparison_result,
                'sources': [],
                'workflow': 'comparison_completed',
                'circular1': circular1_number,
                'circular2': circular2_number,
                'success': True
            }

        except Exception as e:
            return {
                'answer': f"Error processing comparison: {str(e)}",
                'sources': [],
                'workflow': 'comparison_error',
                'success': False
            }


def process_comparison_workflow_simplified(query: str, embedding_service, llm) -> Dict[str, Any]:
    """Simplified comparison workflow"""
    try:
        comparator = CircularComparator(embedding_service, llm)
        result = comparator.process_comparison_query(query)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'answer': f"Error processing comparison: {str(e)}",
                'sources': [],
                'workflow': 'comparison_error',
                'success': False
            })
        }


def initialize_services():
    """Initialize LLM and embedding services"""
    global llm, embedding_service

    if embedding_service is None:
        milvus_config = {
            'host': os.environ['MILVUS_HOST'],
            'port': int(os.environ['MILVUS_PORT']),
            'collection_name': os.environ.get('MILVUS_COLLECTION', 'rag_collection'),
            'connection_args': {
                "host": os.environ['MILVUS_HOST'],
                "port": int(os.environ['MILVUS_PORT'])
            }
        }

        embedding_service = EmbeddingService(
            use_remote=True,
            milvus_config=None,
            use_tei=True,
            endpoint_name=os.environ['SAGEMAKER_ENDPOINT_NAME'],
            region_name=os.environ['IRELAND_REGION'],
            max_batch_size=8
        )

        from milvus_provider.mivlus_provider import MilvusManager
        from pymilvus import utility

        host = milvus_config.get('host', 'localhost')
        port = milvus_config.get('port', '19530')
        collection_name = milvus_config.get('collection_name', 'cssf_documents_final_final')
        connection_args = milvus_config.get('connection_args', {"host": host, "port": int(port)})

        embedding_service.milvus = MilvusManager(
            connection_args=connection_args,
            collection_name=collection_name,
            host=host,
            port=port
        )

        try:
            if utility.has_collection(collection_name):
                embedding_service.milvus.create_collection(embedding_service.provider)
            else:
                embedding_service.milvus.create_collection_with_schema(
                    embedding_provider=embedding_service.provider,
                    vector_dim=768
                )
        except Exception:
            embedding_service.milvus.create_collection(embedding_service.provider)

    if llm is None:
        llm = BedrockLLM(
            model_id="meta.llama3-70b-instruct-v1:0",
            region_name=os.environ['BEDROCK_REGION'],
            model_kwargs={
                "temperature": 0.1,
                "max_tokens": 2000,
                "top_p": 0.9,
                "stop": []
            }
        )


def extract_content_from_response(response: Any) -> str:
    """Extract text content from LLM response object"""
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    elif isinstance(response, str):
        return response
    else:
        return str(response)


def convert_to_langchain_documents(similar_docs: List[Any]) -> List[Document]:
    """Convert embedding service results to LangChain Document objects"""
    documents = []
    for doc_data in similar_docs:
        if isinstance(doc_data, tuple):
            doc_content, score = doc_data
            metadata = doc_content.get('metadata', {})
            metadata['score'] = score
            page_content = doc_content.get('content', doc_content.get('page_content', ''))
        else:
            page_content = doc_data.get('content', doc_data.get('page_content', ''))
            metadata = doc_data.get('metadata', {})

        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    return documents


def rank_documents_by_relevance(query: str, documents: List[Document]) -> List[int]:
    """Get document ranking from LLM"""
    ranking_prompt = f"""Rank these document snippets by relevance to: {query}

Return ONLY a JSON array like: [3, 1, 7, 2, 5]

Snippets:
""" + "\n".join([f"{i + 1}. {doc.page_content[:200]}" for i, doc in enumerate(documents[:10])])

    try:
        ranking_response = llm.invoke(ranking_prompt)
        content = extract_content_from_response(ranking_response)

        # Extract JSON array
        json_match = re.search(r'\[[\d\s,]+\]', content)
        if json_match:
            ranking = json.loads(json_match.group())
            if isinstance(ranking, list):
                return ranking[:10]
    except Exception:
        pass

    return list(range(1, min(11, len(documents) + 1)))


def enhanced_process_query_action(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Enhanced query processing with comparison detection"""
    try:
        query_lower = query.lower()
        comparison_keywords = ['compare', 'comparison', 'versus', 'vs', 'difference between', 'contrast']

        is_comparison = any(keyword in query_lower for keyword in comparison_keywords)

        if is_comparison:
            return process_comparison_workflow_simplified(query, embedding_service, llm)
        else:
            return process_query_action(query, history)

    except Exception:
        return process_query_action(query, history)


def process_query_action(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Main function to process query action"""
    try:
        similar_docs = embedding_service.search_similar_texts(
            query_text=query,
            top_k=20,
            with_scores=True
        )

        documents = convert_to_langchain_documents(similar_docs)

        if not documents:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps({
                    'answer': "No relevant documents found for your query.",
                    'sources': [],
                    'success': True
                })
            }

        ranked_indices = rank_documents_by_relevance(query, documents)

        # Select top documents
        top_docs = []
        for i in ranked_indices:
            if 1 <= i <= len(documents):
                top_docs.append(documents[i - 1])
            if len(top_docs) >= 10:
                break

        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Generate answer
        final_prompt = f"""You are a specialized assistant for Luxembourg and European financial regulations. 
Base your answers on the provided regulatory context.

Context:
{context}

Question: {query}

Answer:"""

        response = llm.invoke(final_prompt)
        answer_text = extract_content_from_response(response)

        # Extract source URLs
        unique_urls = list({
            doc.metadata.get('source_url')
            for doc in top_docs
            if 'source_url' in doc.metadata and doc.metadata.get('source_url')
        })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'answer': answer_text,
                'sources': unique_urls,
                'success': True
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e),
                'success': False
            })
        }


def lambda_handler(event, context):
    """AWS Lambda handler"""
    initialize_services()

    try:
        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '')
        action = body.get('action', 'query')

        if action == 'query':
            if not query:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                    },
                    'body': json.dumps({
                        'error': 'Query is required',
                        'success': False
                    })
                }

            return enhanced_process_query_action(query, [])

        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps({
                    'error': 'Invalid action',
                    'success': False
                })
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e),
                'success': False
            })
        }


if __name__ == "__main__":
    import os

    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'embedding-endpoint'
    os.environ['MILVUS_HOST'] = '34.241.177.15'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['MILVUS_COLLECTION'] = 'cssf_documents_final_final'
    os.environ['BEDROCK_REGION'] = 'us-east-1'
    os.environ['IRELAND_REGION'] = 'eu-west-1'

    test_event = {
        "body": json.dumps({
            "query": "compare circular 16/635 and circular 12/539",
            "action": "query"
        })
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))