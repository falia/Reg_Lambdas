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
    def __init__(self, embedding_service, llm):
        self.embedding_service = embedding_service
        self.llm = llm

    def find_circular_by_number(self, circular_number: str) -> List[Dict]:
        try:
            normalized_number = self._normalize_circular_number(circular_number)
            results = self.find_circular_by_number_adaptive(normalized_number)
            if results:
                return results

            semantic_results = self.embedding_service.search_similar_texts(
                query_text=f"CSSF circular {normalized_number}",
                top_k=200,
                with_scores=True
            )

            filtered_results = []
            for doc_data in semantic_results:
                if isinstance(doc_data, tuple):
                    doc_content, _ = doc_data
                else:
                    doc_content = doc_data

                metadata = doc_content.get('metadata', {})
                doc_type = metadata.get('document_type', '')
                doc_number = metadata.get('document_number', '')
                doc_id = metadata.get('doc_id', '')
                title = metadata.get('title', '')

                type_match = (doc_type == "CSSF circular" or 'CSSF' in title or 'circular' in title.lower())
                number_match = (doc_number == normalized_number or doc_id == normalized_number or normalized_number in title or normalized_number.replace('-', '/') in title)

                if type_match and number_match:
                    filtered_results.append(doc_content)

            return filtered_results

        except Exception:
            return []

    def process_comparison_query(self, query: str) -> Dict[str, Any]:
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

            comparison_result = self.compare_two_circulars(circular1_number, circular2_number)

            return {
                'comparison': comparison_result,
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

    def compare_two_circulars(self, circular1_number: str, circular2_number: str) -> str:
        try:
            chunks1 = self.find_circular_by_number(circular1_number)
            chunks2 = self.find_circular_by_number(circular2_number)

            if not chunks1:
                return f"Could not find circular {circular1_number} in the database."
            if not chunks2:
                return f"Could not find circular {circular2_number} in the database."

            content1 = self._extract_content_from_chunks(chunks1)
            content2 = self._extract_content_from_chunks(chunks2)
            title1 = self._get_title_from_chunks(chunks1) or circular1_number
            title2 = self._get_title_from_chunks(chunks2) or circular2_number

            analysis1 = self._analyze_circular_content(content1, title1)
            analysis2 = self._analyze_circular_content(content2, title2)

            return self._compare_analyses(analysis1, analysis2, title1, title2)

        except Exception as e:
            return f"Error comparing circulars: {str(e)}"

    def _extract_content_from_chunks(self, chunks: List[Dict]) -> str:
        parts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content') or chunk.get('text') or chunk.get('page_content') or ''
                if content:
                    parts.append(content)
        return '\n\n'.join(parts)

    def _get_title_from_chunks(self, chunks: List[Dict]) -> str:
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if 'title' in metadata:
                return metadata['title']
        return None

    def _analyze_circular_content(self, content: str, title: str) -> str:
        try:
            chunk_size = 1500
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            key_findings = []

            for i, chunk in enumerate(chunks[:3]):
                prompt = f"""
Analyze this section of CSSF circular \"{title}\":

{chunk}

Extract key information:
- Main requirements or obligations
- Important definitions
- Key dates or deadlines
- Regulatory status

Be concise (max 3 bullet points):
"""
                try:
                    response = self.llm.invoke(prompt)
                    key_findings.append(f"Section {i+1}: {extract_content_from_response(response).strip()}")
                except Exception:
                    continue

            synthesis_prompt = f"""
Summarize the key aspects of circular \"{title}\" based on these analyses:

{chr(10).join(key_findings)}

Summarize:
- Purpose and scope
- Key requirements
- Current status

Limit to 200 words.
"""
            response = self.llm.invoke(synthesis_prompt)
            return extract_content_from_response(response)

        except Exception:
            return f"Analysis failed for {title}"

    def _compare_analyses(self, analysis1: str, analysis2: str, title1: str, title2: str) -> str:
        prompt = f"""
        Compare these two CSSF circulars and return a structured JSON response with these fields:

        - purpose_and_scope
        - key_differences
        - timeline_and_amendments
        - institutional_impact
        - recommendations

        Use this format:
        {{
          "purpose_and_scope": "...",
          "key_differences": "...",
          "timeline_and_amendments": "...",
          "institutional_impact": "...",
          "recommendations": "..."
        }}

        **{title1} Analysis:**
        {analysis1}

        **{title2} Analysis:**
        {analysis2}
        """

        response = self.llm.invoke(prompt)
        return extract_content_from_response(response)

    def extract_circular_numbers_from_query(self, query: str) -> List[str]:
        patterns = [
            r'CSSF\s+(\d{1,2}[-/]\d{3})',
            r'\b(\d{1,2}[-/]\d{3})\b',
            r'circular\s+(\d{1,2}[-/]\d{3})',
        ]
        found = set()
        query_lower = query.lower()
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                norm = f"CSSF {match}".replace('/', '-') if not match.startswith('CSSF') else match.replace('/', '-')
                found.add(norm)
        return list(found)

    def _normalize_circular_number(self, circular_number: str) -> str:
        number = circular_number.strip().replace('/', '-')
        return number if number.startswith("CSSF") else f"CSSF {number}"

    def find_circular_by_number_adaptive(self, circular_number: str) -> List[Dict]:
        try:
            from pymilvus import Collection
            normalized = self._normalize_circular_number(circular_number)
            collection = Collection(self.embedding_service.milvus.collection_name)
            schema_fields = [f.name for f in collection.schema.fields]

            strategies = []
            if 'document_type' in schema_fields and 'document_number' in schema_fields:
                strategies.append(f'document_type == "CSSF circular" and document_number == "{normalized}"')
            if 'document_number' in schema_fields:
                strategies.append(f'document_number == "{normalized}"')
            if 'doc_id' in schema_fields:
                strategies.append(f'doc_id == "{normalized}"')
            if 'title' in schema_fields:
                for p in [normalized, normalized.replace('-', '/'), normalized.replace('CSSF ', '')]:
                    strategies.append(f'title like "%{p}%"')

            for expr in strategies:
                try:
                    results = collection.query(expr=expr, output_fields=["*"], limit=1000)
                    filtered = []
                    for res in results:
                        title = res.get('title', '')
                        doc_type = res.get('document_type', '')
                        doc_number = res.get('document_number', '')
                        doc_id = res.get('doc_id', '')
                        if ('CSSF' in title or 'circular' in title.lower() or doc_type == 'CSSF circular') and \
                           (doc_number == normalized or doc_id == normalized or normalized in title or normalized.replace('-', '/') in title):
                            filtered.append({
                                'content': res.get('text', ''),
                                'text': res.get('text', ''),
                                'metadata': {
                                    'doc_id': doc_id,
                                    'title': title,
                                    'document_type': doc_type,
                                    'document_number': doc_number,
                                    'lang': res.get('lang', ''),
                                    'url': res.get('url', ''),
                                    'publication_date': res.get('publication_date', '')
                                }
                            })
                    if filtered:
                        return filtered
                except Exception:
                    continue
            return []
        except Exception:
            return []


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
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    elif isinstance(response, str):
        return response
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


def debug_collection_info(embedding_service) -> Dict[str, Any]:
    """Debug function to get collection information"""
    try:
        from pymilvus import Collection, utility

        collection_name = embedding_service.milvus.collection_name

        debug_info = {
            'collection_name': collection_name,
            'collection_exists': utility.has_collection(collection_name),
            'available_collections': utility.list_collections(),
        }

        if debug_info['collection_exists']:
            collection = Collection(collection_name)
            schema = collection.schema

            debug_info.update({
                'is_loaded': collection.is_loaded,
                'num_entities': collection.num_entities,
                'fields': [
                    {
                        'name': field.name,
                        'type': str(field.dtype),
                        'is_primary': field.is_primary,
                        'max_length': getattr(field, 'max_length', None)
                    }
                    for field in schema.fields
                ],
                'field_names': [field.name for field in schema.fields]
            })

            # Try a simple query to test
            try:
                test_results = collection.query(
                    expr="pk >= 0",
                    output_fields=["pk"],
                    limit=1
                )
                debug_info['can_query'] = True
                debug_info['sample_result_keys'] = list(test_results[0].keys()) if test_results else []
            except Exception as e:
                debug_info['can_query'] = False
                debug_info['query_error'] = str(e)

        return debug_info

    except Exception as e:
        return {'error': str(e)}


def lambda_handler(event, context):
    """AWS Lambda handler"""
    initialize_services()

    try:
        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '')
        action = body.get('action', 'query')

        # Debug endpoint
        if action == 'debug':
            debug_info = debug_collection_info(embedding_service)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps(debug_info, indent=2)
            }

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