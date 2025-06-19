import json
import os
import sys
import re
import logging
from typing import List, Dict, Any, Optional
from langchain_aws import BedrockLLM
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from embedding_provider.embedding_provider import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for service reuse
llm = None
embedding_service = None


def initialize_services():
    """Initialize LLM and embedding services (called once per container)"""
    global llm, embedding_service

    if embedding_service is None:
        # Configure your embedding service
        milvus_config = {
            'host': os.environ['MILVUS_HOST'],
            'port': int(os.environ['MILVUS_PORT']),
            'collection_name': os.environ.get('MILVUS_COLLECTION', 'rag_collection'),
            'connection_args': {
                "host": os.environ['MILVUS_HOST'],
                "port": int(os.environ['MILVUS_PORT'])
            }
        }

        # Initialize your embedding service
        embedding_service = EmbeddingService(
            use_remote=True,  # Use SageMaker
            milvus_config=milvus_config,
            use_tei=True,  # Adjust based on your model format
            endpoint_name=os.environ['SAGEMAKER_ENDPOINT_NAME'],
            region_name=os.environ['IRELAND_REGION'],
            max_batch_size=8
        )
        logger.info("Embedding service initialized")

    if llm is None:
        # Initialize Bedrock LLM (Stockholm region)
        llm = BedrockLLM(
            model_id="meta.llama3-70b-instruct-v1:0",  # Adjust model as needed
            region_name=os.environ['BEDROCK_REGION'],
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        )
        logger.info("LLM service initialized")


# =====================================================
# UTILITY FUNCTIONS FOR QUERY PROCESSING
# =====================================================

def convert_to_langchain_documents(similar_docs: List[Any]) -> List[Document]:
    """Convert embedding service results to LangChain Document objects"""
    documents = []
    for doc_data in similar_docs:
        if isinstance(doc_data, tuple):  # with scores
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


def format_snippets_for_ranking(documents: List[Document]) -> str:
    """Format document snippets for LLM ranking"""
    newline_char = '\n'
    space_char = ' '
    return "\n".join([
        f"{i + 1}. {doc.page_content[:1000].replace(newline_char, space_char)}"
        for i, doc in enumerate(documents)
    ])


def create_ranking_prompt(query: str, formatted_snippets: str) -> str:
    """Create an optimized ranking prompt for the LLM"""
    return f"""You are a document ranking assistant. Rank the following document snippets by their relevance to the question.

Question: {query}

Snippets:
{formatted_snippets}

Instructions:
- Return ONLY a JSON array of snippet numbers
- Order from most relevant to least relevant
- Include top 10 most relevant snippets
- Do not include any explanation or additional text

Example format: [3, 1, 7, 2, 5, 8, 4, 6, 9, 10]

Your ranking:"""


def extract_content_from_response(response: Any) -> str:
    """Extract text content from LLM response object"""
    if hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)


def parse_ranking_response(response: Any, max_docs: int) -> List[int]:
    """Extract ranking from LLM response with multiple fallback strategies"""

    content = extract_content_from_response(response)
    logger.info(f"Extracted content: {content}")

    # Strategy 1: Look for JSON array pattern
    json_pattern = r'\[[\d\s,]+\]'
    json_matches = re.findall(json_pattern, content)

    if json_matches:
        try:
            json_str = json_matches[0]
            ranking = json.loads(json_str)
            if isinstance(ranking, list) and all(isinstance(x, int) for x in ranking):
                logger.info(f"Successfully parsed JSON ranking: {ranking}")
                return ranking[:10]
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")

    # Strategy 2: Extract individual numbers
    numbers = re.findall(r'\b\d+\b', content)
    if numbers:
        try:
            ranking = [int(x) for x in numbers if 1 <= int(x) <= max_docs]
            logger.info(f"Extracted numbers ranking: {ranking}")
            return ranking[:10]
        except ValueError:
            pass

    # Strategy 3: Fallback to sequential order
    logger.info("Using fallback sequential order")
    return list(range(1, min(11, max_docs + 1)))


def rank_documents_by_relevance(query: str, documents: List[Document]) -> List[int]:
    """Get document ranking from LLM with robust error handling"""

    formatted_snippets = format_snippets_for_ranking(documents)
    ranking_prompt = create_ranking_prompt(query, formatted_snippets)

    logger.info("Ranking documents...")
    logger.debug(f"Ranking prompt: {ranking_prompt[-500:]}")  # Last 500 chars

    try:
        ranking_response = llm.invoke(ranking_prompt)
        logger.info(f"Ranking response: {ranking_response}")

        ranked_indices = parse_ranking_response(ranking_response, len(documents))
        logger.info(f"Final ranking indices: {ranked_indices}")
        return ranked_indices

    except Exception as e:
        logger.error(f"Failed to rank documents: {e}")
        return list(range(1, min(11, len(documents) + 1)))


def select_top_documents(documents: List[Document], ranked_indices: List[int], max_docs: int = 10) -> List[Document]:
    """Select and reorder documents based on ranking indices"""

    top_docs = []

    # Add documents in ranked order
    for i in ranked_indices:
        if 1 <= i <= len(documents):
            top_docs.append(documents[i - 1])
        if len(top_docs) >= max_docs:
            break

    # Fill with remaining documents if needed
    if len(top_docs) < max_docs:
        used_indices = set(ranked_indices)
        for i in range(1, len(documents) + 1):
            if i not in used_indices and len(top_docs) < max_docs:
                top_docs.append(documents[i - 1])

    logger.info(f"Selected {len(top_docs)} documents for final context")
    return top_docs


def create_context_from_documents(documents: List[Document]) -> str:
    """Create context string from selected documents"""
    return "\n\n".join([doc.page_content for doc in documents])


def create_final_prompt(query: str, context: str) -> str:
    """Create the final prompt for answering the question"""
    return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""


def extract_unique_source_urls(documents: List[Document]) -> List[str]:
    """Extract unique source URLs from document metadata"""
    return list({
        doc.metadata.get('source_url')
        for doc in documents
        if 'source_url' in doc.metadata and doc.metadata.get('source_url')
    })


def generate_answer_with_llm(query: str, context: str) -> str:
    """Generate final answer using LLM"""

    final_prompt = create_final_prompt(query, context)

    logger.info("Generating final answer...")
    logger.info(f"Final prompt: {final_prompt}")

    response = llm.invoke(final_prompt)

    # Extract response content properly
    return extract_content_from_response(response)


# =====================================================
# ACTION PROCESSING FUNCTIONS
# =====================================================

def process_query_action(query: str) -> Dict[str, Any]:
    """Main function to process query action with document ranking"""

    try:
        # Step 1: Retrieve similar documents
        similar_docs = embedding_service.search_similar_texts(
            query_text=query,
            top_k=20,  # retrieve more to let LLM rank
            with_scores=True
        )

        # Step 2: Convert to LangChain documents
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

        # Step 3: Rank documents by relevance
        ranked_indices = rank_documents_by_relevance(query, documents)

        # Step 4: Select top-ranked documents
        top_docs = select_top_documents(documents, ranked_indices, max_docs=10)

        # Step 5: Create context and generate answer
        context = create_context_from_documents(top_docs)

        answer_text = generate_answer_with_llm(query, context)

        # Step 6: Extract source URLs
        unique_urls = extract_unique_source_urls(top_docs)

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
        logger.error(f"Error in process_query_action: {str(e)}")
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


def process_add_documents_action(documents_data: List[Any]) -> Dict[str, Any]:
    """Process add documents action"""
    try:
        # Extract texts and metadata
        texts = []
        metadatas = []

        for doc_data in documents_data:
            if isinstance(doc_data, dict):
                texts.append(doc_data.get('content', doc_data.get('text', '')))
                metadatas.append(doc_data.get('metadata', {}))
            else:
                texts.append(str(doc_data))
                metadatas.append({})

        # Use your embedding service to add documents
        result = embedding_service.add_texts_to_store(texts, metadatas)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'message': f'Successfully added {result["count"]} documents',
                'milvus_ids': result.get('milvus_ids', []),
                'success': True
            })
        }
    except Exception as e:
        logger.error(f"Error in process_add_documents_action: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': 'Failed to add documents',
                'details': str(e),
                'success': False
            })
        }


def process_search_similar_action(query_text: str, top_k: int = 10, with_scores: bool = True) -> Dict[str, Any]:
    """Process search similar action"""
    try:
        results = embedding_service.search_similar_texts(
            query_text=query_text,
            top_k=top_k,
            with_scores=with_scores
        )

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'results': results,
                'success': True
            })
        }
    except Exception as e:
        logger.error(f"Error in process_search_similar_action: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': 'Search failed',
                'details': str(e),
                'success': False
            })
        }


# =====================================================
# LAMBDA HANDLER
# =====================================================

def lambda_handler(event, context):
    """AWS Lambda handler function"""

    # Initialize services on first invocation
    initialize_services()

    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '')
        action = body.get('action', 'query')

        # Route to appropriate action
        if action == 'query':
            if not query:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                    },
                    'body': json.dumps({
                        'error': 'Query is required for action "query"',
                        'success': False
                    })
                }
            return process_query_action(query)

        elif action == 'add_documents':
            documents_data = body.get('documents', [])
            if not documents_data:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                    },
                    'body': json.dumps({
                        'error': 'Documents are required for action "add_documents"',
                        'success': False
                    })
                }
            return process_add_documents_action(documents_data)

        elif action == 'search_similar':
            if not query:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                    },
                    'body': json.dumps({
                        'error': 'Query is required for action "search_similar"',
                        'success': False
                    })
                }

            top_k = body.get('top_k', 10)
            with_scores = body.get('with_scores', True)

            return process_search_similar_action(query, top_k, with_scores)

        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps({
                    'error': 'Invalid action. Use "query", "add_documents", or "search_similar"',
                    'success': False
                })
            }

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'error': 'Invalid JSON in request body',
                'details': str(e),
                'success': False
            })
        }

    except Exception as e:
        import traceback

        try:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_type is not None:
                error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            else:
                error_details = str(e)

            logger.error("🛑 Error in RAG handler:\n" + error_details)
        except Exception as log_err:
            logger.error("Failed to log exception:")
            logger.error(f"Original error: {e}")
            logger.error(f"Logging error: {log_err}")

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


# =====================================================
# LOCAL TESTING
# =====================================================

if __name__ == "__main__":
    import os
    import json

    # Set fake env vars (mimic Lambda environment)
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'embedding-endpoint'
    os.environ['MILVUS_HOST'] = '34.241.177.15'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['MILVUS_COLLECTION'] = 'cssf_documents'
    os.environ['BEDROCK_REGION'] = 'us-east-1'
    os.environ['IRELAND_REGION'] = 'eu-west-1'

    # Test event
    test_event = {
        "body": json.dumps({
            "query": "What is a prospectus?",
            "action": "query"
        })
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))