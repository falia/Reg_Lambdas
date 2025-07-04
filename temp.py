import json
import boto3
import uuid
import time
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from decimal import Decimal
from langchain_aws import BedrockLLM
from langchain_core.documents import Document
from embedding_provider.embedding_provider import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

# Environment variables
JOB_STATUS_TABLE = os.environ.get('JOB_STATUS_TABLE')
JOB_TTL_HOURS = int(os.environ.get('JOB_TTL_HOURS', 24))
FUNCTION_NAME = os.environ.get('AWS_LAMBDA_FUNCTION_NAME')

# DynamoDB table
job_table = dynamodb.Table(JOB_STATUS_TABLE) if JOB_STATUS_TABLE else None

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
                number_match = (
                            doc_number == normalized_number or doc_id == normalized_number or normalized_number in title or normalized_number.replace(
                        '-', '/') in title)

                if type_match and number_match:
                    filtered_results.append(doc_content)

            return filtered_results

        except Exception:
            return []

    def process_comparison_query(self, query: str, job_id: str = None) -> Dict[str, Any]:
        try:
            if job_id:
                update_job_status(job_id, 'PROCESSING', progress='Extracting circular numbers from query...')

            circular_numbers = self.extract_circular_numbers_from_query(query)

            if len(circular_numbers) < 2:
                return {
                    'answer': "I need at least two circular numbers to compare. Please specify both circulars in your query (e.g., 'compare CSSF 16-635 and CSSF 12-539').",
                    'sources': [],
                    'pdfs': [],
                    'workflow': 'comparison_insufficient_numbers',
                    'found_numbers': circular_numbers
                }

            circular1_number = circular_numbers[0]
            circular2_number = circular_numbers[1]

            if job_id:
                update_job_status(job_id, 'PROCESSING',
                                  progress=f'Comparing {circular1_number} and {circular2_number}...')

            comparison_result = self.compare_two_circulars(circular1_number, circular2_number, job_id)

            if 'error' in comparison_result:
                return {
                    'answer': comparison_result['error'],
                    'sources': comparison_result.get('sources', []),
                    'pdfs': comparison_result.get('pdfs', []),
                    'workflow': 'comparison_error',
                    'circular1': circular1_number,
                    'circular2': circular2_number,
                    'success': False
                }

            return {
                'comparison': comparison_result['comparison'],
                'sources': comparison_result.get('sources', []),
                'pdfs': comparison_result.get('pdfs', []),
                'workflow': 'comparison_completed',
                'circular1': circular1_number,
                'circular2': circular2_number,
                'circular1_title': comparison_result.get('circular1_title'),
                'circular2_title': comparison_result.get('circular2_title'),
                'success': True
            }

        except Exception as e:
            return {
                'answer': f"Error processing comparison: {str(e)}",
                'sources': [],
                'pdfs': [],
                'workflow': 'comparison_error',
                'success': False
            }

    def compare_two_circulars(self, circular1_number: str, circular2_number: str, job_id: str = None) -> Dict[str, Any]:
        try:
            if job_id:
                update_job_status(job_id, 'PROCESSING', progress=f'Finding circular {circular1_number}...')

            chunks1 = self.find_circular_by_number(circular1_number)

            if job_id:
                update_job_status(job_id, 'PROCESSING', progress=f'Finding circular {circular2_number}...')

            chunks2 = self.find_circular_by_number(circular2_number)

            if not chunks1:
                return {
                    'error': f"Could not find circular {circular1_number} in the database.",
                    'pdfs': []
                }
            if not chunks2:
                return {
                    'error': f"Could not find circular {circular2_number} in the database.",
                    'pdfs': []
                }

            content1 = self._extract_content_from_chunks(chunks1)
            content2 = self._extract_content_from_chunks(chunks2)
            title1 = self._get_title_from_chunks(chunks1) or circular1_number
            title2 = self._get_title_from_chunks(chunks2) or circular2_number

            # Extract both PDF URLs and source URLs from chunk metadata
            all_chunks = chunks1 + chunks2
            pdf_urls = self._extract_pdf_urls_from_chunks(all_chunks)
            source_urls = self._extract_source_urls_from_chunks(all_chunks)

            if job_id:
                update_job_status(job_id, 'PROCESSING', progress=f'Analyzing {title1}...')

            analysis1 = self._analyze_circular_content(content1, title1)

            if job_id:
                update_job_status(job_id, 'PROCESSING', progress=f'Analyzing {title2}...')

            analysis2 = self._analyze_circular_content(content2, title2)

            if job_id:
                update_job_status(job_id, 'PROCESSING', progress='Generating comparison...')

            comparison_result = self._compare_analyses(analysis1, analysis2, title1, title2)

            return {
                'comparison': comparison_result,
                'pdfs': pdf_urls,
                'sources': source_urls,
                'circular1_title': title1,
                'circular2_title': title2
            }

        except Exception as e:
            return {
                'error': f"Error comparing circulars: {str(e)}",
                'pdfs': [],
                'sources': []
            }

    def _extract_content_from_chunks(self, chunks: List[Dict]) -> str:
        parts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get('content') or chunk.get('text') or chunk.get('page_content') or ''
                if content:
                    parts.append(content)
        return '\n\n'.join(parts)

    def _extract_source_urls_from_chunks(self, chunks: List[Dict]) -> List[str]:
        """Extract unique source URLs from chunk metadata top_related field"""
        source_urls = []
        seen_urls = set()

        for chunk in chunks:
            metadata = chunk.get('metadata', {})

            # Extract URLs only from top_related field
            top_related = metadata.get('top_related', [])
            if isinstance(top_related, list):
                for related_item in top_related:
                    if isinstance(related_item, dict) and 'url' in related_item:
                        url = related_item['url']
                        if url and url not in seen_urls:
                            source_urls.append(url)
                            seen_urls.add(url)
                            logger.info(f"Found source URL in top_related: {url}")

            # Limit to prevent too many URLs
            if len(source_urls) >= 20:
                break

        logger.info(f"Extracted {len(source_urls)} source URLs from {len(chunks)} chunks")
        return source_urls

    def _extract_pdf_urls_from_chunks(self, chunks: List[Dict]) -> List[Dict[str, str]]:
        """Extract unique PDF URLs from chunk metadata"""
        pdf_urls = []
        seen_urls = set()

        for chunk in chunks:
            metadata = chunk.get('metadata', {})

            # Extract PDFs from top_related field (main structure)
            top_related = metadata.get('top_related', [])
            if isinstance(top_related, list):
                for related_item in top_related:
                    if isinstance(related_item, dict) and 'url' in related_item:
                        url = related_item['url']
                        content_type = related_item.get('content_type', '')

                        # Check if it's a PDF
                        if url and url not in seen_urls:
                            if (url.lower().endswith('.pdf') or
                                    'pdf' in url.lower() or
                                    content_type == 'application/pdf'):

                                # Extract title from URL (fallback)
                                title = url.split('/')[-1].replace('.pdf', '').replace('_', ' ').title()

                                pdf_info = {
                                    'url': url,
                                    'title': title,
                                    'content_type': content_type,
                                    's3_uri': related_item.get('s3_uri', ''),
                                    'circular_number': metadata.get('document_number', ''),
                                    'document_type': metadata.get('document_type', ''),
                                    'publication_date': metadata.get('publication_date', ''),
                                    'crawl_session': metadata.get('crawl_session', '')
                                }
                                pdf_urls.append(pdf_info)
                                seen_urls.add(url)

                                # Limit to prevent too many URLs
                                if len(pdf_urls) >= 10:
                                    break

            if len(pdf_urls) >= 10:
                break

        return pdf_urls

    def _get_title_from_chunks(self, chunks: List[Dict]) -> str:
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if 'title' in metadata:
                return metadata['title']
        return None

    def _analyze_circular_content(self, content: str, title: str) -> str:
        try:
            chunk_size = 1500
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
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
                    key_findings.append(f"Section {i + 1}: {extract_content_from_response(response).strip()}")
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
                                (
                                        doc_number == normalized or doc_id == normalized or normalized in title or normalized.replace(
                                    '-', '/') in title):
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


def process_rag_query(job_id: str, request_body: Dict[str, Any]) -> None:
    """Process the RAG query with your existing logic"""
    try:
        # Initialize services
        initialize_services()

        # Update status to processing
        update_job_status(job_id, 'PROCESSING', progress='Initializing RAG processing...')

        query = request_body.get('query', '')
        max_results = request_body.get('max_results', 20)
        threshold = request_body.get('threshold', 0.7)

        # Check if this is a comparison query
        query_lower = query.lower()
        comparison_keywords = ['compare', 'comparison', 'versus', 'vs', 'difference between', 'contrast']
        is_comparison = any(keyword in query_lower for keyword in comparison_keywords)

        if is_comparison:
            # Process comparison query
            update_job_status(job_id, 'PROCESSING', progress='Processing comparison query...')

            comparator = CircularComparator(embedding_service, llm)
            result = comparator.process_comparison_query(query, job_id)

            # Update job status to completed
            update_job_status(job_id, 'COMPLETED', result=result)

        else:
            # Process regular RAG query
            update_job_status(job_id, 'PROCESSING', progress='Searching for relevant documents...')

            similar_docs = embedding_service.search_similar_texts(
                query_text=query,
                top_k=max_results,
                with_scores=True
            )

            documents = convert_to_langchain_documents(similar_docs)

            if not documents:
                result = {
                    'answer': "No relevant documents found for your query.",
                    'sources': [],
                    'success': True
                }
                update_job_status(job_id, 'COMPLETED', result=result)
                return

            update_job_status(job_id, 'PROCESSING', progress='Ranking documents by relevance...')

            ranked_indices = rank_documents_by_relevance(query, documents)

            # Select top documents
            top_docs = []
            for i in ranked_indices:
                if 1 <= i <= len(documents):
                    top_docs.append(documents[i - 1])
                if len(top_docs) >= 10:
                    break

            update_job_status(job_id, 'PROCESSING', progress='Generating answer...')

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

            # Extract source URLs only from top_related
            source_urls = []
            seen_urls = set()

            for doc in top_docs:
                metadata = doc.metadata

                # Extract URLs only from top_related field
                top_related = metadata.get('top_related', [])
                if isinstance(top_related, list):
                    for related_item in top_related:
                        if isinstance(related_item, dict) and 'url' in related_item:
                            url = related_item['url']
                            if url and url not in seen_urls:
                                source_urls.append(url)
                                seen_urls.add(url)

            unique_urls = source_urls[:20]  # Limit to 20 URLs

            result = {
                'answer': answer_text,
                'sources': unique_urls,
                'success': True,
                'query': query,
                'num_documents_found': len(documents),
                'num_documents_used': len(top_docs)
            }

            # Update job status to completed
            update_job_status(job_id, 'COMPLETED', result=result)

        logger.info(f"Successfully processed RAG query for job {job_id}")

    except Exception as e:
        logger.error(f"Error processing RAG query for job {job_id}: {str(e)}")
        update_job_status(job_id, 'FAILED', error=str(e))


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for async RAG processing
    """
    # Check if this is a direct Lambda invocation (async processing)
    if 'action' in event and event['action'] == 'PROCESS_RAG':
        job_id = event.get('jobId')
        request_body = event.get('requestBody', {})

        if job_id and request_body:
            process_rag_query(job_id, request_body)
        else:
            logger.error("Invalid direct invocation payload")

        return {'statusCode': 200}

    # Otherwise, handle as API Gateway request
    try:
        # Extract HTTP method and path
        http_method = event.get('httpMethod', '')
        path = event.get('path', '')

        logger.info(f"Processing {http_method} request to {path}")

        # Route based on path and method
        if path == '/health' and http_method == 'GET':
            return handle_health_check()
        elif path == '/query' and http_method == 'POST':
            return handle_query_submission(event)
        elif path.startswith('/status/') and http_method == 'GET':
            job_id = path.split('/')[-1]
            return handle_status_check(job_id)
        elif path == '/debug' and http_method == 'GET':
            return handle_debug_request(event)
        elif path == '/env' and http_method == 'GET':
            return handle_env_debug()
        elif path == '/metadata' and http_method == 'GET':
            return handle_metadata_debug()
        else:
            return create_response(404, {'error': 'Not Found'})

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return create_response(500, {'error': 'Internal Server Error'})


def handle_health_check() -> Dict[str, Any]:
    """Handle health check requests"""
    return create_response(200, {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': os.environ.get('VERSION_TIMESTAMP', 'unknown')
    })


def decimal_default(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def convert_decimals(obj):
    """Convert DynamoDB Decimal objects to float for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(v) for v in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


def handle_query_submission(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle query submission and start async processing"""
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '').strip()

        if not query:
            return create_response(400, {'error': 'Query is required'})

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Calculate TTL (time to live) for the job
        ttl = int((datetime.utcnow() + timedelta(hours=JOB_TTL_HOURS)).timestamp())

        # Store initial job status
        job_item = {
            'jobId': job_id,
            'status': 'PENDING',
            'query': query,
            'createdAt': datetime.utcnow().isoformat(),
            'updatedAt': datetime.utcnow().isoformat(),
            'ttl': ttl
        }

        # Add optional parameters (convert floats to Decimal for DynamoDB)
        if 'max_results' in body:
            job_item['maxResults'] = Decimal(str(body['max_results']))
        if 'threshold' in body:
            job_item['threshold'] = Decimal(str(body['threshold']))
        if 'action' in body:
            job_item['action'] = body['action']

        # Store job in DynamoDB
        if job_table:
            job_table.put_item(Item=job_item)

        # Start async processing
        start_async_processing(job_id, body)

        return create_response(202, {
            'jobId': job_id,
            'status': 'PENDING',
            'message': 'Query submitted successfully. Use the jobId to check status.',
            'statusUrl': f"/status/{job_id}"
        })

    except json.JSONDecodeError:
        return create_response(400, {'error': 'Invalid JSON in request body'})
    except Exception as e:
        logger.error(f"Error submitting query: {str(e)}")
        return create_response(500, {'error': 'Failed to submit query'})


def handle_status_check(job_id: str) -> Dict[str, Any]:
    """Handle status check requests"""
    try:
        if not job_table:
            return create_response(500, {'error': 'Job status table not configured'})

        # Get job status from DynamoDB
        response = job_table.get_item(Key={'jobId': job_id})

        if 'Item' not in response:
            return create_response(404, {'error': 'Job not found'})

        job = response['Item']

        # Convert Decimals to floats for JSON serialization
        job = convert_decimals(job)

        # Prepare response
        status_response = {
            'jobId': job_id,
            'status': job['status'],
            'createdAt': job['createdAt'],
            'updatedAt': job['updatedAt']
        }

        # Add query for reference
        if 'query' in job:
            status_response['query'] = job['query']

        # Add result if completed
        if job['status'] == 'COMPLETED' and 'result' in job:
            status_response['result'] = job['result']

        # Add error if failed
        if job['status'] == 'FAILED' and 'error' in job:
            status_response['error'] = job['error']

        # Add progress if processing
        if job['status'] == 'PROCESSING' and 'progress' in job:
            status_response['progress'] = job['progress']

        return create_response(200, status_response)

    except Exception as e:
        logger.error(f"Error checking status for job {job_id}: {str(e)}")
        return create_response(500, {'error': 'Failed to check job status'})


def start_async_processing(job_id: str, request_body: Dict[str, Any]) -> None:
    """Start asynchronous processing by invoking this Lambda function"""
    try:
        payload = {
            'jobId': job_id,
            'requestBody': request_body,
            'action': 'PROCESS_RAG'
        }

        # Invoke Lambda function asynchronously
        lambda_client.invoke(
            FunctionName=FUNCTION_NAME,
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps(payload)
        )

        logger.info(f"Started async processing for job {job_id}")

    except Exception as e:
        logger.error(f"Failed to start async processing for job {job_id}: {str(e)}")
        # Update job status to failed
        update_job_status(job_id, 'FAILED', error=str(e))


def update_job_status(job_id: str, status: str, result: Optional[Dict] = None,
                      error: Optional[str] = None, progress: Optional[str] = None) -> None:
    """Update job status in DynamoDB"""
    try:
        if not job_table:
            return

        update_expression = "SET #status = :status, updatedAt = :updated_at"
        expression_attribute_names = {'#status': 'status'}
        expression_attribute_values = {
            ':status': status,
            ':updated_at': datetime.utcnow().isoformat()
        }

        if result:
            update_expression += ", #result = :result"
            expression_attribute_names['#result'] = 'result'
            expression_attribute_values[':result'] = result

        if error:
            update_expression += ", #error = :error"
            expression_attribute_names['#error'] = 'error'
            expression_attribute_values[':error'] = error

        if progress:
            update_expression += ", progress = :progress"
            expression_attribute_values[':progress'] = progress

        job_table.update_item(
            Key={'jobId': job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )

    except Exception as e:
        logger.error(f"Failed to update job status for {job_id}: {str(e)}")


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create HTTP response with CORS headers"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
        },
        'body': json.dumps(body, default=decimal_default)
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


def handle_metadata_debug() -> Dict[str, Any]:
    """Debug metadata fields available in vector database"""
    try:
        initialize_services()

        # Search for a sample of documents
        sample_docs = embedding_service.search_similar_texts(
            query_text="CSSF",
            top_k=5,
            with_scores=True
        )

        metadata_info = {
            'total_docs_sampled': len(sample_docs),
            'sample_metadata_fields': [],
            'field_examples': {}
        }

        for i, doc_data in enumerate(sample_docs[:3]):
            if isinstance(doc_data, tuple):
                doc_content, score = doc_data
            else:
                doc_content = doc_data

            metadata = doc_content.get('metadata', {})

            metadata_info['sample_metadata_fields'].append({
                f'doc_{i}': list(metadata.keys())
            })

            # Collect examples of each field
            for field, value in metadata.items():
                if field not in metadata_info['field_examples']:
                    metadata_info['field_examples'][field] = []
                if len(metadata_info['field_examples'][field]) < 3:
                    metadata_info['field_examples'][field].append(str(value)[:100])

        return create_response(200, metadata_info)

    except Exception as e:
        return create_response(500, {'error': f'Metadata debug failed: {str(e)}'})


def handle_env_debug() -> Dict[str, Any]:
    """Debug environment variables and configuration"""
    try:
        env_info = {
            'JOB_STATUS_TABLE': os.environ.get('JOB_STATUS_TABLE'),
            'AWS_LAMBDA_FUNCTION_NAME': os.environ.get('AWS_LAMBDA_FUNCTION_NAME'),
            'SAGEMAKER_ENDPOINT_NAME': os.environ.get('SAGEMAKER_ENDPOINT_NAME'),
            'MILVUS_HOST': os.environ.get('MILVUS_HOST'),
            'MILVUS_PORT': os.environ.get('MILVUS_PORT'),
            'MILVUS_COLLECTION': os.environ.get('MILVUS_COLLECTION'),
            'BEDROCK_REGION': os.environ.get('BEDROCK_REGION'),
            'IRELAND_REGION': os.environ.get('IRELAND_REGION'),
            'job_table_configured': job_table is not None,
            'lambda_client_configured': lambda_client is not None,
        }

        # Test DynamoDB table access
        if job_table:
            try:
                # Try to describe the table
                table_info = job_table.table_status
                env_info['dynamodb_table_status'] = table_info
            except Exception as e:
                env_info['dynamodb_error'] = str(e)

        return create_response(200, env_info)
    except Exception as e:
        return create_response(500, {'error': f'Environment debug failed: {str(e)}'})


# Additional debug endpoint for testing
def handle_debug_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle debug requests for testing purposes"""
    try:
        initialize_services()
        debug_info = debug_collection_info(embedding_service)
        return create_response(200, debug_info)
    except Exception as e:
        return create_response(500, {'error': f'Debug failed: {str(e)}'})


# Test function for local development
if __name__ == "__main__":
    import os

    # Set test environment variables
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'embedding-endpoint'
    os.environ['MILVUS_HOST'] = '34.241.177.15'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['MILVUS_COLLECTION'] = 'cssf_documents_final_final'
    os.environ['BEDROCK_REGION'] = 'us-east-1'
    os.environ['IRELAND_REGION'] = 'eu-west-1'
    os.environ['JOB_STATUS_TABLE'] = 'test-job-status'
    os.environ['AWS_LAMBDA_FUNCTION_NAME'] = 'test-function'

    # Test comparison query
    test_event = {
        "httpMethod": "POST",
        "path": "/query",
        "body": json.dumps({
            "query": "compare circular 16/635 and circular 12/539",
            "action": "query"
        })
    }

    result = lambda_handler(test_event, None)
    print("Query submission result:")
    print(json.dumps(result, indent=2))

    # Test regular query
    test_event_regular = {
        "httpMethod": "POST",
        "path": "/query",
        "body": json.dumps({
            "query": "What are the requirements for risk management?",
            "max_results": 10
        })
    }

    result_regular = lambda_handler(test_event_regular, None)
    print("\nRegular query submission result:")
    print(json.dumps(result_regular, indent=2))

    # Test health check
    health_event = {
        "httpMethod": "GET",
        "path": "/health"
    }

    health_result = lambda_handler(health_event, None)
    print("\nHealth check result:")
    print(json.dumps(health_result, indent=2))