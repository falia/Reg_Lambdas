# Enhanced RAG system with simplified circular comparison
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import os
import sys
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


# =====================================================
# SIMPLIFIED CIRCULAR COMPARISON
# =====================================================

class CircularComparator:
    """Simplified circular comparison using only metadata fields"""

    def __init__(self, embedding_service, llm):
        self.embedding_service = embedding_service
        self.llm = llm

    def find_circular_by_number(self, circular_number: str) -> List[Dict]:
        """
        Find all chunks for a specific circular using direct metadata search
        Args:
            circular_number: e.g., "CSSF 12-539" or "12/539" 
        """
        try:
            # Normalize the circular number to match your metadata format
            normalized_number = self._normalize_circular_number(circular_number)

            logger.info(f"🔍 Searching for circular with document_number: '{normalized_number}'")

            # Since you have the fields, try direct query on the collection
            try:
                # Use pymilvus directly to query the collection
                from pymilvus import Collection

                # Get the collection
                collection = Collection(self.embedding_service.milvus.collection_name)

                # Build the query expression
                query_expr = f'document_type == "CSSF circular" and document_number == "{normalized_number}"'

                logger.info(f"🔍 Query expression: {query_expr}")

                # Execute the query
                results = collection.query(
                    expr=query_expr,
                    output_fields=["*"],  # Get all fields
                    limit=1000
                )

                if results:
                    # Convert results to the expected format
                    formatted_results = []
                    for result in results:
                        # Create document format
                        doc = {
                            'content': result.get('text', ''),
                            'text': result.get('text', ''),
                            'metadata': {
                                'doc_id': result.get('doc_id', ''),
                                'title': result.get('title', ''),
                                'document_type': result.get('document_type', ''),
                                'document_number': result.get('document_number', ''),
                                'lang': result.get('lang', ''),
                                'super_category': result.get('super_category', ''),
                                'subtitle': result.get('subtitle', ''),
                                'url': result.get('url', ''),
                                'publication_date': result.get('publication_date', ''),
                                'update_date': result.get('update_date', ''),
                                'content_hash': result.get('content_hash', ''),
                                'entities_text': result.get('entities_text', ''),
                                'keywords_text': result.get('keywords_text', ''),
                                'themes_text': result.get('themes_text', ''),
                            }
                        }
                        formatted_results.append(doc)

                    logger.info(
                        f"✅ Found {len(formatted_results)} chunks for circular {normalized_number} via direct query")
                    return formatted_results

                else:
                    logger.warning(f"❌ No chunks found for circular {normalized_number} via direct query")
                    return []

            except Exception as direct_query_error:
                logger.error(f"❌ Direct query failed: {direct_query_error}")
                logger.info("🔄 Falling back to semantic search + filtering")

                # Fallback to semantic search + filtering
                results = self.embedding_service.search_similar_texts(
                    query_text=f"CSSF circular {normalized_number}",
                    top_k=200,
                    with_scores=True
                )

                # Filter results
                filtered_results = []
                for doc_data in results:
                    if isinstance(doc_data, tuple):
                        doc_content, score = doc_data
                    else:
                        doc_content = doc_data

                    metadata = doc_content.get('metadata', {})
                    doc_type = metadata.get('document_type', '')
                    doc_number = metadata.get('document_number', '')
                    doc_id = metadata.get('doc_id', '')
                    title = metadata.get('title', '')

                    if (doc_type == "CSSF circular" and
                            (doc_number == normalized_number or
                             doc_id == normalized_number or
                             normalized_number in title)):
                        filtered_results.append(doc_content)

                if filtered_results:
                    logger.info(
                        f"✅ Found {len(filtered_results)} chunks for circular {normalized_number} via semantic search fallback")
                    return filtered_results
                else:
                    logger.warning(f"❌ No chunks found for circular {normalized_number}")
                    return []

        except Exception as e:
            logger.error(f"❌ Error searching for circular {circular_number}: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return []

    def _normalize_circular_number(self, circular_number: str) -> str:
        """
        Normalize circular number to match your metadata format
        Convert formats like "16/635" to "CSSF 16-635" or keep "CSSF 12-539" as is
        """
        # Remove any whitespace
        number = circular_number.strip()

        # If it already starts with "CSSF", return as is
        if number.startswith("CSSF"):
            return number

        # Convert slash to dash and add CSSF prefix
        if "/" in number:
            number = number.replace("/", "-")

        # Add CSSF prefix if not present
        if not number.startswith("CSSF"):
            number = f"CSSF {number}"

        return number

    def extract_circular_numbers_from_query(self, query: str) -> List[str]:
        """
        Extract circular numbers from query using simple regex patterns
        """
        import re

        # Look for patterns like "16/635", "12-539", "CSSF 12-539", etc.
        patterns = [
            r'CSSF\s+(\d{1,2}[-/]\d{3})',  # "CSSF 12-539" or "CSSF 16/635"
            r'\b(\d{1,2}[-/]\d{3})\b',  # "12-539" or "16/635"
            r'circular\s+(\d{1,2}[-/]\d{3})',  # "circular 16/635"
        ]

        found_numbers = []
        query_lower = query.lower()

        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                # Add CSSF prefix if not present
                if not match.startswith('CSSF'):
                    normalized = f"CSSF {match}".replace('/', '-')
                else:
                    normalized = match.replace('/', '-')

                if normalized not in found_numbers:
                    found_numbers.append(normalized)

        logger.info(f"📋 Extracted circular numbers: {found_numbers}")
        return found_numbers

    def compare_two_circulars(self, circular1_number: str, circular2_number: str, query: str) -> str:
        """
        Compare two circulars by their document numbers
        """
        try:
            # Get all chunks for both circulars
            circular1_chunks = self.find_circular_by_number(circular1_number)
            circular2_chunks = self.find_circular_by_number(circular2_number)

            # Check if we found both circulars
            if not circular1_chunks:
                return f"❌ Could not find circular {circular1_number} in the database. Please check the circular number."

            if not circular2_chunks:
                return f"❌ Could not find circular {circular2_number} in the database. Please check the circular number."

            # Extract content from all chunks
            content1 = self._extract_content_from_chunks(circular1_chunks)
            content2 = self._extract_content_from_chunks(circular2_chunks)

            # Get titles for reference
            title1 = self._get_title_from_chunks(circular1_chunks) or circular1_number
            title2 = self._get_title_from_chunks(circular2_chunks) or circular2_number

            logger.info(
                f"🔄 Comparing {title1} ({len(circular1_chunks)} chunks) vs {title2} ({len(circular2_chunks)} chunks)")

            # Generate comparison using LLM
            comparison_result = self._generate_comparison(content1, content2, title1, title2, query)

            return comparison_result

        except Exception as e:
            logger.error(f"❌ Error comparing circulars: {e}")
            return f"❌ Error occurred while comparing circulars: {str(e)}"

    def _extract_content_from_chunks(self, chunks: List[Dict]) -> str:
        """Extract and combine content from all chunks"""
        content_parts = []

        logger.info(f"📄 Extracting content from {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                # Try different possible content fields
                content = (chunk.get('content') or
                           chunk.get('text') or
                           chunk.get('page_content') or
                           '')

                if content:
                    content_parts.append(content)
                    logger.info(f"📄 Chunk {i + 1}: {len(content)} chars")
                else:
                    logger.warning(f"📄 Chunk {i + 1}: No content found")
                    # Debug: show chunk keys
                    chunk_keys = list(chunk.keys())
                    logger.info(f"📄 Chunk {i + 1} keys: {chunk_keys}")

        combined_content = '\n\n'.join(content_parts)
        logger.info(f"📄 Combined content: {len(combined_content)} chars")

        return combined_content

    def _get_title_from_chunks(self, chunks: List[Dict]) -> str:
        """Extract title from chunk metadata"""
        for chunk in chunks:
            if isinstance(chunk, dict):
                metadata = chunk.get('metadata', {})
                title = metadata.get('title')
                if title:
                    return title
        return None

    def _generate_comparison(self, content1: str, content2: str, title1: str, title2: str, original_query: str) -> str:
        """Generate comparison using LLM"""

        # Debug content lengths
        logger.info(f"📏 Content lengths: Circular 1: {len(content1)} chars, Circular 2: {len(content2)} chars")

        # Truncate content if too long (keep first part of each circular)
        max_content_length = 3000  # Reduce to avoid content filtering
        if len(content1) > max_content_length:
            content1 = content1[:max_content_length] + "\n\n[Content truncated...]"
            logger.info(f"📏 Truncated content1 to {max_content_length} chars")
        if len(content2) > max_content_length:
            content2 = content2[:max_content_length] + "\n\n[Content truncated...]"
            logger.info(f"📏 Truncated content2 to {max_content_length} chars")

        # Ensure we have content
        if not content1.strip():
            content1 = f"[No content available for {title1}]"
        if not content2.strip():
            content2 = f"[No content available for {title2}]"

        # Try a simpler prompt first to test LLM connectivity
        simple_test_prompt = f"""
        You are a regulatory expert. Compare these two CSSF circulars briefly:

        Circular 1: {title1}
        Circular 2: {title2}

        Please provide a brief comparison in 2-3 sentences.
        """

        logger.info(f"📝 Testing LLM with simple prompt ({len(simple_test_prompt)} chars)...")

        try:
            # Test with simple prompt first
            test_response = self.llm.invoke(simple_test_prompt)
            test_content = extract_content_from_response(test_response)

            if test_content.strip():
                logger.info(f"📝 Simple test successful: {len(test_content)} chars")
                # If simple test works, try the full comparison
                return self._generate_full_comparison(content1, content2, title1, title2, original_query)
            else:
                logger.error("❌ Simple test failed - LLM returned empty response")
                return self._generate_fallback_comparison(content1, content2, title1, title2)

        except Exception as e:
            logger.error(f"❌ Error in simple test: {e}")
            return self._generate_fallback_comparison(content1, content2, title1, title2)

    def _generate_full_comparison(self, content1: str, content2: str, title1: str, title2: str,
                                  original_query: str) -> str:
        """Generate full comparison with detailed prompt"""

        comparison_prompt = f"""
        You are a financial regulatory expert. Compare these two CSSF circulars:

        **CIRCULAR 1: {title1}**
        {content1[:2000]}

        **CIRCULAR 2: {title2}**
        {content2[:2000]}

        Provide a structured comparison:

        ## Executive Summary
        Brief overview of what each circular covers and their relationship.

        ## Key Differences
        - Main differences between the circulars
        - Changed requirements or procedures
        - New or removed obligations

        ## Impact Assessment
        What these changes mean for financial institutions.

        ## Conclusion
        Overall significance of the changes.

        Keep the response focused and concise.
        """

        logger.info(f"📝 Generating full comparison ({len(comparison_prompt)} chars)...")

        try:
            response = self.llm.invoke(comparison_prompt)
            extracted_content = extract_content_from_response(response)

            if extracted_content.strip():
                logger.info(f"📝 Full comparison successful: {len(extracted_content)} chars")
                return extracted_content
            else:
                logger.error("❌ Full comparison failed - empty response")
                return self._generate_fallback_comparison(content1, content2, title1, title2)

        except Exception as e:
            logger.error(f"❌ Error in full comparison: {e}")
            return self._generate_fallback_comparison(content1, content2, title1, title2)

    def _generate_fallback_comparison(self, content1: str, content2: str, title1: str, title2: str) -> str:
        """Generate a fallback comparison when LLM fails"""

        logger.info("📝 Generating fallback comparison...")

        # Basic text analysis
        content1_words = len(content1.split())
        content2_words = len(content2.split())

        # Look for common regulatory terms
        regulatory_terms = ['article', 'section', 'requirement', 'obligation', 'compliance', 'procedure', 'shall',
                            'must']

        content1_terms = sum(1 for term in regulatory_terms if term.lower() in content1.lower())
        content2_terms = sum(1 for term in regulatory_terms if term.lower() in content2.lower())

        fallback_comparison = f"""
## Circular Comparison: {title1} vs {title2}

**Note:** This is a basic comparison due to LLM processing limitations.

### Document Analysis
- **{title1}**: {content1_words} words, {content1_terms} regulatory terms found
- **{title2}**: {content2_words} words, {content2_terms} regulatory terms found

### Basic Observations
- Circular 1 appears to be {"longer" if content1_words > content2_words else "shorter"} than Circular 2
- Both circulars contain regulatory language and compliance requirements
- The titles suggest these circulars may have a related regulatory purpose

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

**Note:** This comparison is limited due to automated processing constraints. A professional regulatory analysis is recommended for compliance purposes.
"""

        return fallback_comparison

    def process_comparison_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to process circular comparison queries
        """
        try:
            # Extract circular numbers from query
            circular_numbers = self.extract_circular_numbers_from_query(query)

            if len(circular_numbers) < 2:
                return {
                    'answer': "❌ I need at least two circular numbers to compare. Please specify both circulars in your query (e.g., 'compare CSSF 16-635 and CSSF 12-539').",
                    'sources': [],
                    'workflow': 'comparison_insufficient_numbers',
                    'found_numbers': circular_numbers
                }

            # Take first two circular numbers
            circular1_number = circular_numbers[0]
            circular2_number = circular_numbers[1]

            logger.info(f"🔄 Processing comparison: {circular1_number} vs {circular2_number}")

            # Perform comparison
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
            logger.error(f"❌ Error processing comparison query: {e}")
            return {
                'answer': f"❌ Error processing comparison: {str(e)}",
                'sources': [],
                'workflow': 'comparison_error',
                'success': False
            }


def process_comparison_workflow_simplified(query: str, embedding_service, llm) -> Dict[str, Any]:
    """
    Simplified comparison workflow
    """
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
        logger.error(f"❌ Error in simplified comparison workflow: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
            },
            'body': json.dumps({
                'answer': f"❌ Error processing comparison: {str(e)}",
                'sources': [],
                'workflow': 'comparison_error',
                'success': False
            })
        }


# =====================================================
# EXISTING UTILITY FUNCTIONS
# =====================================================

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

        # Initialize your embedding service WITHOUT automatic collection creation
        embedding_service = EmbeddingService(
            use_remote=True,  # Use SageMaker
            milvus_config=None,  # Don't auto-create collection here
            use_tei=True,  # Adjust based on your model format
            endpoint_name=os.environ['SAGEMAKER_ENDPOINT_NAME'],
            region_name=os.environ['IRELAND_REGION'],
            max_batch_size=8
        )

        # Now manually set up Milvus with the CORRECT method
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

        # Check if collection exists and handle schema mismatch
        try:
            if utility.has_collection(collection_name):
                logger.info(f"🔍 Collection '{collection_name}' already exists")

                # Try to use existing collection with simple method (for compatibility)
                logger.info("🔧 Using existing collection with simple schema...")
                embedding_service.milvus.create_collection(embedding_service.provider)
                logger.info("✅ Connected to existing Milvus collection")
            else:
                # Collection doesn't exist, create with custom schema
                logger.info("🔧 Creating new Milvus collection with custom schema...")
                embedding_service.milvus.create_collection_with_schema(
                    embedding_provider=embedding_service.provider,
                    vector_dim=768
                )
                logger.info("✅ Created new Milvus collection with custom schema")

        except Exception as schema_error:
            logger.error(f"❌ Schema error: {schema_error}")

            # If there's a schema mismatch, use the simple method as fallback
            logger.info("🔄 Schema mismatch detected, using simple collection method...")
            try:
                embedding_service.milvus.create_collection(embedding_service.provider)
                logger.info("✅ Connected to existing collection with simple method")
            except Exception as fallback_error:
                logger.error(f"❌ Failed to connect with simple method: {fallback_error}")
                raise Exception(f"Could not initialize Milvus collection: {fallback_error}")

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
    logger.info(f"📝 Extracting content from response type: {type(response)}")

    if hasattr(response, 'content'):
        content = response.content
        logger.info(f"📝 Found .content attribute: {len(content)} chars")
        return content
    elif hasattr(response, 'text'):
        content = response.text
        logger.info(f"📝 Found .text attribute: {len(content)} chars")
        return content
    elif hasattr(response, 'body'):
        content = response.body
        logger.info(f"📝 Found .body attribute: {len(content)} chars")
        return content
    elif isinstance(response, str):
        logger.info(f"📝 Response is already string: {len(response)} chars")
        return response
    else:
        # Log all attributes for debugging
        attrs = [attr for attr in dir(response) if not attr.startswith('_')]
        logger.info(f"📝 Response attributes: {attrs}")

        # Try to convert to string
        content = str(response)
        logger.info(f"📝 Converted to string: {len(content)} chars")
        return content


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
    logger.debug(f"Ranking prompt: {ranking_prompt[-500:]}")

    try:
        ranking_response = llm.invoke(ranking_prompt)
        logger.debug(f"Ranking response: {ranking_response}")

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


def create_final_prompt(query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Create the final prompt for answering the question with optional conversation history"""
    prompt_parts = [
        "You are a specialized assistant for Luxembourg and European financial regulations. "
        "Your role is to provide accurate, compliance-focused answers based on official regulatory documents.",
        "",
        "IMPORTANT GUIDELINES:",
        "- Base your answers strictly on the provided regulatory context",
        "- If information is not available in the context, clearly state this limitation",
        "- For regulatory requirements, cite specific articles, sections, or provisions when available",
        "- Distinguish between EU-wide regulations and Luxembourg-specific requirements",
        "- Use precise regulatory terminology (e.g., 'prospectus', 'competent authority', 'home Member State')",
        "- If asked about compliance obligations, emphasize consulting official sources or legal counsel",
        "- Do not provide general financial advice - focus on regulatory information only",
        ""
    ]

    # Add conversation history if available
    if history and len(history) > 0:
        prompt_parts.append("Previous conversation context:")
        for i, exchange in enumerate(history[-5:], 1):
            user_msg = exchange.get('user', exchange.get('question', ''))
            assistant_msg = exchange.get('assistant', exchange.get('answer', ''))
            if user_msg and assistant_msg:
                prompt_parts.append(f"{i}. User: {user_msg}")
                prompt_parts.append(f"   Assistant: {assistant_msg}")
        prompt_parts.append("")

    # Add current context and question
    prompt_parts.extend([
        "Regulatory Context:",
        context,
        "",
        f"Question: {query}",
        "",
        "Answer (based on the regulatory context provided):"
    ])

    return "\n".join(prompt_parts)


def extract_unique_source_urls(documents: List[Document]) -> List[str]:
    """Extract unique source URLs from document metadata"""
    return list({
        doc.metadata.get('source_url')
        for doc in documents
        if 'source_url' in doc.metadata and doc.metadata.get('source_url')
    })


def generate_answer_with_llm(query: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Generate final answer using LLM with optional conversation history"""
    final_prompt = create_final_prompt(query, context, history)

    logger.info("Generating final answer...")
    logger.debug(f"Final prompt: {final_prompt[:500]}...")

    response = llm.invoke(final_prompt)
    return extract_content_from_response(response)


def validate_history_format(history: List[Any]) -> List[Dict[str, str]]:
    """Validate and normalize conversation history format"""
    validated_history = []

    for exchange in history:
        if isinstance(exchange, dict):
            user_msg = exchange.get('user') or exchange.get('question') or exchange.get('human')
            assistant_msg = exchange.get('assistant') or exchange.get('answer') or exchange.get('ai')

            if user_msg and assistant_msg:
                validated_history.append({
                    'user': str(user_msg),
                    'assistant': str(assistant_msg)
                })
        elif isinstance(exchange, (list, tuple)) and len(exchange) >= 2:
            validated_history.append({
                'user': str(exchange[0]),
                'assistant': str(exchange[1])
            })

    return validated_history


# =====================================================
# ENHANCED QUERY PROCESSING
# =====================================================

def enhanced_process_query_action(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Enhanced query processing with simplified comparison detection"""

    try:
        # Simple comparison detection using keywords
        query_lower = query.lower()
        comparison_keywords = ['compare', 'comparison', 'versus', 'vs', 'difference between', 'contrast']

        # Check if this is a comparison query
        is_comparison = any(keyword in query_lower for keyword in comparison_keywords)

        if is_comparison:
            logger.info("🔄 Detected comparison query, using simplified workflow")
            return process_comparison_workflow_simplified(query, embedding_service, llm)
        else:
            logger.info("🔄 Using standard query processing")
            return process_query_action(query, history)

    except Exception as e:
        logger.error(f"❌ Error in enhanced query processing: {e}")
        return process_query_action(query, history)  # Fallback to original


def process_query_action(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Main function to process query action with document ranking and conversation history"""
    try:
        # Step 1: Retrieve similar documents
        similar_docs = embedding_service.search_similar_texts(
            query_text=query,
            top_k=20,
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

        # Step 5: Create context and generate answer with history
        context = create_context_from_documents(top_docs)
        answer_text = generate_answer_with_llm(query, context, history)

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


# =====================================================
# EXISTING FUNCTIONS
# =====================================================

def process_add_documents_action(documents_data: List[Any]) -> Dict[str, Any]:
    """Process add documents action"""
    try:
        texts = []
        metadatas = []

        for doc_data in documents_data:
            if isinstance(doc_data, dict):
                texts.append(doc_data.get('content', doc_data.get('text', '')))
                metadatas.append(doc_data.get('metadata', {}))
            else:
                texts.append(str(doc_data))
                metadatas.append({})

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
    """AWS Lambda handler with simplified circular comparison"""

    # Initialize services on first invocation
    initialize_services()

    try:
        # Parse request
        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '')
        action = body.get('action', 'query')
        history = body.get('history', [])

        # Validate and normalize history if provided
        validated_history = []
        if history and isinstance(history, list):
            validated_history = validate_history_format(history)
            logger.info(f"Processed {len(validated_history)} history exchanges")

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

            logger.info(f"🚀 Lambda handler processing query: '{query}'")
            return enhanced_process_query_action(query, validated_history)

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

            logger.error("Error in RAG handler:\n" + error_details)
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

    # Set real env vars (mimic Lambda environment)
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'embedding-endpoint'
    os.environ['MILVUS_HOST'] = '34.241.177.15'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['MILVUS_COLLECTION'] = 'cssf_documents_final_final'
    os.environ['BEDROCK_REGION'] = 'us-east-1'
    os.environ['IRELAND_REGION'] = 'eu-west-1'

    # Test full system with existing circulars
    test_event = {
        "body": json.dumps({
            "query": "compare circular 16/635 and circular 12/539",  # These exist in the database
            "action": "query"
        })
    }

    print("\nTesting full system with existing circulars...")
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))