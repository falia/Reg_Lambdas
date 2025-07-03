# Enhanced RAG system with intent detection and circular comparison
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
# AGENTIC AI COMPONENTS
# =====================================================

class QueryIntent(Enum):
    """Define different types of query intents"""
    GENERAL_QUERY = "general_query"
    COMPARE_CIRCULARS = "compare_circulars"
    FIND_DIFFERENCES = "find_differences"
    ANALYZE_CHANGES = "analyze_changes"
    SUMMARIZE_CIRCULAR = "summarize_circular"
    COMPLIANCE_CHECK = "compliance_check"


class AgentState(Enum):
    """Track agent's current state"""
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    COMPARING = "comparing"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


class IntentDetector:
    """Detect user intent from query"""

    def __init__(self, llm):
        self.llm = llm

    def detect_intent(self, query: str) -> Tuple[QueryIntent, Dict[str, Any]]:
        """Detect intent and extract parameters from user query"""

        # Simple keyword-based detection first (faster and more reliable)
        intent_keywords = {
            QueryIntent.COMPARE_CIRCULARS: [
                "compare", "comparison", "versus", "vs", "difference between",
                "contrast", "compare circulars", "compare between", "différence entre"
            ],
            QueryIntent.FIND_DIFFERENCES: [
                "differences", "what changed", "what's different", "changes between",
                "modifications", "updates", "revisions", "changements"
            ],
            QueryIntent.ANALYZE_CHANGES: [
                "analyze changes", "impact of changes", "what does this change mean",
                "implications", "effects of", "analyse", "impact", "analyze", "analyse"
            ],
            QueryIntent.SUMMARIZE_CIRCULAR: [
                "summarize", "summary", "key points", "main points", "overview",
                "résumé", "synthèse"
            ],
            QueryIntent.COMPLIANCE_CHECK: [
                "compliance", "requirements", "obligations", "must", "should",
                "conformité", "exigences", "obligations"
            ]
        }

        query_lower = query.lower()

        # Check for keyword matches first
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                params = self._extract_parameters(query, intent)
                logger.info(f"Keyword-based intent detection: {intent.value}")
                return intent, params

        # Only use LLM if keywords don't match AND query is complex enough
        if len(query.split()) > 3:  # Only for longer queries
            logger.info("Using LLM intent detection for complex query")
            return self._llm_intent_detection(query)
        else:
            # For short queries, default to general
            logger.info("Short query, defaulting to general intent")
            return QueryIntent.GENERAL_QUERY, {}

    def _extract_parameters(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """Extract parameters with backup extraction prioritized for circular comparisons"""
        params = {}

        if intent in [QueryIntent.COMPARE_CIRCULARS, QueryIntent.FIND_DIFFERENCES, QueryIntent.ANALYZE_CHANGES]:

            # For circular comparisons, try backup extraction first since it's more reliable
            if intent == QueryIntent.COMPARE_CIRCULARS:
                logger.info("Using backup extraction for circular comparison (more reliable)")
                params = self._backup_circular_extraction(query)

                # If backup found circular numbers, we're good to go
                if params.get('circulars'):
                    logger.info(f"Backup extraction successful: {params}")
                    return params

            # If no circular numbers found or for other intents, try LLM
            logger.info("Trying LLM extraction")
            extraction_prompt = f"""
            Extract information from: "{query}"

            Return only JSON:
            {{"circular_identifiers": [], "topic_area": "", "confidence": 0.9}}
            """

            try:
                response = self.llm.invoke(extraction_prompt)
                content = extract_content_from_response(response).strip()
                content = self._clean_json_response(content)

                logger.info(f"LLM response: {content}")

                extracted_data = json.loads(content)

                # Map extracted data to parameters
                extracted_circulars = extracted_data.get('circular_identifiers', [])
                params['circulars'] = extracted_circulars[:2]
                params['specific_focus'] = extracted_data.get('specific_focus')
                params['temporal_context'] = extracted_data.get('temporal_context', '')
                params['topic_area'] = extracted_data.get('topic_area', '')
                params['confidence'] = extracted_data.get('confidence', 0.5)

                logger.info(f"LLM extracted parameters: {params}")

            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
                # Always fall back to backup extraction
                params = self._backup_circular_extraction(query)
                logger.info(f"Using backup extraction as fallback: {params}")

        return params

    def _backup_circular_extraction(self, query: str) -> Dict[str, Any]:
        """Backup extraction using simple text patterns when LLM fails"""
        params = {'circulars': []}

        # Simple text patterns to find circular numbers
        import re

        # Look for patterns like "16/635", "12/539", etc.
        number_patterns = [
            r'\b(\d{1,2}/\d{3})\b',  # "16/635", "12/539"
            r'\b(\d{1,2}-\d{3})\b',  # "16-635", "12-539"
            r'\b(\d{4}/\d{3})\b',  # "2023/635"
        ]

        found_numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, query)
            found_numbers.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_numbers = []
        for num in found_numbers:
            if num not in seen:
                seen.add(num)
                unique_numbers.append(num)

        params['circulars'] = unique_numbers[:2]  # Limit to 2

        # Also extract basic topic keywords
        query_lower = query.lower()
        topic_keywords = {
            'AML': ['aml', 'anti-money laundering'],
            'MiFID': ['mifid'],
            'UCITS': ['ucits'],
            'prospectus': ['prospectus', 'prospectuses'],
            'governance': ['governance'],
            'risk': ['risk management']
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                params['topic_area'] = topic
                break

        params['confidence'] = 0.7 if params['circulars'] else 0.3

        logger.info(f"Backup extraction result: {params}")
        return params

    def _minimal_fallback_extraction(self, query: str) -> Dict[str, Any]:
        """Minimal fallback extraction using only topic keywords - NO REGEX for circular numbers"""
        params = {'circulars': []}  # Empty - let the search handle it differently

        # Only extract basic topic keywords
        query_lower = query.lower()
        topic_keywords = {
            'AML': ['aml', 'anti-money laundering', 'money laundering'],
            'MiFID': ['mifid', 'markets in financial instruments'],
            'UCITS': ['ucits', 'undertakings for collective investment'],
            'risk management': ['risk management', 'risk'],
            'governance': ['governance', 'corporate governance'],
            'outsourcing': ['outsourcing', 'third party'],
            'prospectus': ['prospectus', 'prospectuses', 'securities']
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                params['topic_area'] = topic
                break

        # Basic temporal indicators
        if any(word in query_lower for word in ['latest', 'new', 'recent', 'current']):
            params['temporal_context'] = 'latest_vs_previous'
        elif any(word in query_lower for word in ['previous', 'old', 'former']):
            params['temporal_context'] = 'new_vs_old'

        params['confidence'] = 0.3  # Low confidence for fallback

        return params

    def _llm_intent_detection(self, query: str) -> Tuple[QueryIntent, Dict[str, Any]]:
        """Use LLM to detect intent when keywords don't match"""

        intent_prompt = f"""
        Analyze this user query and determine the intent. You MUST respond with ONLY a valid JSON object, no other text.

        Query: "{query}"

        Possible intents:
        - general_query: General questions about regulations/documents
        - compare_circulars: User wants to compare two or more circulars
        - find_differences: User wants to find what changed between versions
        - analyze_changes: User wants to understand the impact of changes
        - summarize_circular: User wants a summary of a circular
        - compliance_check: User wants to check compliance requirements

        Respond with ONLY this JSON format (no markdown, no explanation):
        {{"intent": "general_query", "confidence": 0.8, "parameters": {{"circulars": [], "specific_focus": ""}}}}
        """

        try:
            response = self.llm.invoke(intent_prompt)
            content = extract_content_from_response(response).strip()

            # Clean up common JSON formatting issues
            content = self._clean_json_response(content)

            logger.info(f"LLM response for intent detection: {content}")

            # Parse JSON response
            result = json.loads(content)
            intent_str = result.get('intent', 'general_query')

            # Convert string to enum
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                logger.warning(f"Unknown intent: {intent_str}, defaulting to general_query")
                intent = QueryIntent.GENERAL_QUERY

            params = result.get('parameters', {})
            confidence = result.get('confidence', 0.5)

            logger.info(f"Detected intent: {intent.value} with confidence: {confidence}")
            return intent, params

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in LLM intent detection: {e}")
            logger.error(f"Raw LLM response: {content}")
            return QueryIntent.GENERAL_QUERY, {}

        except Exception as e:
            logger.error(f"LLM intent detection failed: {e}")
            return QueryIntent.GENERAL_QUERY, {}

    def _clean_json_response(self, content: str) -> str:
        """Clean up common JSON formatting issues from LLM responses"""
        try:
            # Remove markdown code blocks
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])  # Remove first and last lines

            # Remove any "json" language identifier
            content = content.replace('```json', '').replace('```', '')

            # Strip whitespace
            content = content.strip()

            # Find JSON object boundaries
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx + 1]

            return content

        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return content


class CircularComparator:
    """Handle circular comparison workflows"""

    def __init__(self, embedding_service, llm):
        self.embedding_service = embedding_service
        self.llm = llm

    def find_circulars_by_identifiers(self, identifiers: List[str]) -> List[Dict]:
        """Find circulars by their identifiers/numbers using flexible LLM-cleaned identifiers with extensive debugging"""
        found_circulars = []

        for identifier in identifiers:
            logger.info(f"🔍 SEARCHING for circular with LLM-extracted identifier: '{identifier}'")

            # Search using metadata with document_type filter
            try:
                results = []

                # Strategy 1: Test if we can find ANY CSSF circulars first
                logger.info("📊 Testing basic CSSF circular search...")
                if hasattr(self.embedding_service, 'milvus_manager'):
                    test_results = self.embedding_service.milvus_manager.search_by_metadata(
                        document_type="CSSF circular",
                        limit=5
                    )
                    logger.info(f"📈 Found {len(test_results) if test_results else 0} CSSF circulars in total")

                    if test_results:
                        # Log sample titles to see what's available
                        for i, doc in enumerate(test_results[:3]):
                            metadata = doc.get('metadata', {})
                            title = metadata.get('title', 'No title')
                            doc_number = metadata.get('document_number', 'No doc_number')
                            doc_id = metadata.get('doc_id', 'No doc_id')
                            logger.info(
                                f"📄 Sample {i + 1}: title='{title}', doc_number='{doc_number}', doc_id='{doc_id}'")

                # Strategy 2: Direct searches with identifier variations
                if hasattr(self.embedding_service, 'milvus_manager'):
                    # Try different title formats that might contain this identifier
                    title_variations = [
                        identifier,  # "16/635"
                        f"CSSF {identifier}",  # "CSSF 16/635"
                        f"Circular CSSF {identifier}",  # "Circular CSSF 16/635"
                        f"Circular {identifier}",  # "Circular 16/635"
                        f"{identifier.replace('/', '//')}"  # Handle double slashes
                    ]

                    for title_variant in title_variations:
                        if results:
                            break
                        logger.info(f"🔎 Trying title search with: '{title_variant}'")
                        search_results = self.embedding_service.milvus_manager.search_by_metadata(
                            document_type="CSSF circular",
                            title_contains=title_variant,
                            limit=50
                        )
                        if search_results:
                            results.extend(search_results)
                            logger.info(f"✅ Found {len(search_results)} results with title variant: '{title_variant}'")
                        else:
                            logger.info(f"❌ No results for title variant: '{title_variant}'")

                # Strategy 3: Search by document number field
                if not results and hasattr(self.embedding_service, 'milvus_manager'):
                    logger.info(f"🔎 Trying document_number search with: '{identifier}'")
                    search_results = self.embedding_service.milvus_manager.search_by_metadata(
                        document_type="CSSF circular",
                        document_number=identifier,
                        limit=50
                    )
                    if search_results:
                        results.extend(search_results)
                        logger.info(f"✅ Found {len(search_results)} results with document_number")
                    else:
                        logger.info(f"❌ No results for document_number: '{identifier}'")

                # Strategy 4: Search by doc_id field
                if not results and hasattr(self.embedding_service, 'milvus_manager'):
                    logger.info(f"🔎 Trying doc_id search with: '{identifier}'")
                    search_results = self.embedding_service.milvus_manager.search_by_metadata(
                        document_type="CSSF circular",
                        doc_id=identifier,
                        limit=50
                    )
                    if search_results:
                        results.extend(search_results)
                        logger.info(f"✅ Found {len(search_results)} results with doc_id")
                    else:
                        logger.info(f"❌ No results for doc_id: '{identifier}'")

                # Strategy 5: Broad semantic search with filtering
                if not results:
                    logger.info(f"🔎 Trying broad semantic search...")
                    semantic_queries = [
                        identifier,  # Just the number
                        f"CSSF circular {identifier}",
                        f"Circular CSSF {identifier}",
                        f"circular {identifier}",
                        f"CSSF {identifier}",
                        f"16/635" if identifier == "16/635" else identifier,  # Hardcode test
                        f"12/539" if identifier == "12/539" else identifier  # Hardcode test
                    ]

                    for semantic_query in semantic_queries:
                        if results:
                            break
                        logger.info(f"🔎 Trying semantic search with: '{semantic_query}'")

                        try:
                            semantic_results = self.embedding_service.search_similar_texts(
                                query_text=semantic_query,
                                top_k=20,
                                with_scores=True
                            )

                            logger.info(
                                f"📊 Semantic search returned {len(semantic_results) if semantic_results else 0} total results")

                            # Filter by document_type and check for identifier presence
                            cssf_count = 0
                            for doc_data in semantic_results:
                                if isinstance(doc_data, tuple):
                                    doc_content, score = doc_data
                                else:
                                    doc_content = doc_data
                                    score = 0

                                metadata = doc_content.get('metadata', {})
                                doc_type = metadata.get('document_type', 'Unknown')

                                if doc_type == 'CSSF circular':
                                    cssf_count += 1
                                    # Check if the identifier appears anywhere in the metadata
                                    title = metadata.get('title', '').lower()
                                    doc_number = metadata.get('document_number', '').lower()
                                    doc_id = metadata.get('doc_id', '').lower()

                                    identifier_lower = identifier.lower()

                                    # Log what we're comparing
                                    logger.info(
                                        f"🔍 Checking: identifier='{identifier_lower}' vs title='{title}', doc_number='{doc_number}', doc_id='{doc_id}', score={score}")

                                    if (identifier_lower in title or
                                            identifier_lower in doc_number or
                                            identifier_lower in doc_id or
                                            score > 0.7):  # High semantic similarity
                                        results.append(doc_content)
                                        logger.info(f"✅ MATCH FOUND! Added document with score={score}")

                            logger.info(f"📊 Found {cssf_count} CSSF circulars in semantic results")

                            if results:
                                logger.info(
                                    f"✅ Found {len(results)} matching results with semantic search: '{semantic_query}'")

                        except Exception as semantic_error:
                            logger.error(f"❌ Semantic search failed for '{semantic_query}': {semantic_error}")

                if results:
                    # Smart chunk processing for large circulars
                    processed_results = self._process_circular_chunks(results, identifier)
                    found_circulars.extend(processed_results)
                    logger.info(
                        f"🎉 Successfully found {len(processed_results)} processed chunks for identifier: {identifier}")
                else:
                    logger.warning(f"⚠️ NO CSSF circulars found for LLM-extracted identifier: {identifier}")

                    # Final debug: Try a completely different approach
                    logger.info("🔍 Final debug attempt - searching for ANY documents with this identifier...")
                    try:
                        debug_results = self.embedding_service.search_similar_texts(
                            query_text=identifier,
                            top_k=10,
                            with_scores=True
                        )

                        if debug_results:
                            logger.info(f"📊 Debug search found {len(debug_results)} total documents")
                            for i, doc_data in enumerate(debug_results[:3]):
                                if isinstance(doc_data, tuple):
                                    doc_content, score = doc_data
                                else:
                                    doc_content = doc_data
                                    score = 0

                                metadata = doc_content.get('metadata', {})
                                doc_type = metadata.get('document_type', 'Unknown')
                                title = metadata.get('title', 'No title')

                                logger.info(
                                    f"📄 Debug result {i + 1}: type='{doc_type}', title='{title}', score={score}")
                        else:
                            logger.info("📊 Debug search found NO documents at all")

                    except Exception as debug_error:
                        logger.error(f"❌ Debug search failed: {debug_error}")

            except Exception as e:
                logger.error(f"❌ Error searching for circular {identifier}: {e}")
                import traceback
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")

        logger.info(f"🏁 FINAL RESULT: Found {len(found_circulars)} total circular chunks")
        return found_circulars

    def _process_circular_chunks(self, chunks: List[Dict], identifier: str) -> List[Dict]:
        """
        Smart processing of circular chunks to optimize for comparison
        """
        try:
            # Group chunks by document (same doc_id)
            doc_groups = {}
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                doc_id = metadata.get('doc_id', 'unknown')

                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(chunk)

            processed_results = []

            for doc_id, doc_chunks in doc_groups.items():
                # Sort chunks by page number if available
                sorted_chunks = self._sort_chunks_by_page(doc_chunks)

                # Strategy based on number of chunks
                if len(sorted_chunks) <= 5:
                    # Small circular: include all chunks
                    processed_results.extend(sorted_chunks)
                    logger.info(f"Small circular {doc_id}: including all {len(sorted_chunks)} chunks")

                elif len(sorted_chunks) <= 15:
                    # Medium circular: include key sections + sample content
                    key_chunks = self._select_key_chunks(sorted_chunks, target_count=10)
                    processed_results.extend(key_chunks)
                    logger.info(f"Medium circular {doc_id}: selected {len(key_chunks)} key chunks")

                else:
                    # Large circular (20+ pages): intelligent summarization
                    summarized_content = self._create_circular_summary(sorted_chunks, doc_id)
                    processed_results.append(summarized_content)
                    logger.info(f"Large circular {doc_id}: created summary from {len(sorted_chunks)} chunks")

            return processed_results

        except Exception as e:
            logger.error(f"Error processing circular chunks: {e}")
            return chunks  # Return original chunks if processing fails

    def _sort_chunks_by_page(self, chunks: List[Dict]) -> List[Dict]:
        """Sort chunks by page number if available"""

        def get_page_number(chunk):
            metadata = chunk.get('metadata', {})
            page_num = metadata.get('page_number', 0)
            return int(page_num) if page_num else 0

        try:
            return sorted(chunks, key=get_page_number)
        except Exception as e:
            logger.error(f"Error sorting chunks by page: {e}")
            return chunks

    def _select_key_chunks(self, chunks: List[Dict], target_count: int = 10) -> List[Dict]:
        """
        Select the most important chunks from a medium-sized circular
        """
        try:
            # Prioritize chunks with key regulatory content
            key_indicators = [
                'requirements', 'obligations', 'compliance', 'procedures',
                'definitions', 'scope', 'application', 'sanctions',
                'reporting', 'governance', 'risk management', 'controls'
            ]

            scored_chunks = []
            for chunk in chunks:
                content = chunk.get('content', chunk.get('text', ''))

                # Calculate relevance score based on key indicators
                score = 0
                content_lower = content.lower()

                for indicator in key_indicators:
                    if indicator in content_lower:
                        score += 1

                # Boost score for chunks with numbered sections/articles
                if any(pattern in content for pattern in ['article', 'section', 'chapter', 'point']):
                    score += 2

                # Boost score for chunks with specific regulatory language
                if any(word in content_lower for word in ['shall', 'must', 'required', 'prohibited']):
                    score += 1

                scored_chunks.append((chunk, score))

            # Sort by score (descending) and take top chunks
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Ensure we get a good spread across the document
            selected_chunks = []
            selected_pages = set()

            for chunk, score in scored_chunks:
                if len(selected_chunks) >= target_count:
                    break

                page_num = chunk.get('metadata', {}).get('page_number', 0)

                # Include high-scoring chunks or chunks from new pages
                if score >= 2 or page_num not in selected_pages:
                    selected_chunks.append(chunk)
                    selected_pages.add(page_num)

            # Fill remaining slots with highest-scoring chunks
            remaining_slots = target_count - len(selected_chunks)
            for chunk, score in scored_chunks:
                if remaining_slots <= 0:
                    break
                if chunk not in selected_chunks:
                    selected_chunks.append(chunk)
                    remaining_slots -= 1

            return selected_chunks[:target_count]

        except Exception as e:
            logger.error(f"Error selecting key chunks: {e}")
            return chunks[:target_count]

    def _create_circular_summary(self, chunks: List[Dict], doc_id: str) -> Dict:
        """
        Create an intelligent summary of a large circular using LLM
        """
        try:
            # Extract key sections and content
            all_content = []
            metadata_sample = {}

            for chunk in chunks:
                content = chunk.get('content', chunk.get('text', ''))
                if content:
                    all_content.append(content)

                # Keep metadata from first chunk
                if not metadata_sample:
                    metadata_sample = chunk.get('metadata', {})

            # Combine content with smart truncation
            combined_content = '\n\n'.join(all_content)

            # If content is very long, use LLM to create structured summary
            if len(combined_content) > 10000:
                summary_prompt = f"""
                Create a comprehensive structured summary of this CSSF circular for comparison purposes.
                Focus on key regulatory requirements, obligations, and compliance aspects.

                Circular Content:
                {combined_content[:8000]}...

                Provide a structured summary with:
                1. **Purpose & Scope**: What this circular covers
                2. **Key Requirements**: Main obligations and requirements
                3. **Compliance Obligations**: What institutions must do
                4. **Important Definitions**: Key terms and definitions
                5. **Deadlines & Timelines**: Any important dates
                6. **Penalties & Sanctions**: Consequences for non-compliance

                Keep the summary comprehensive but focused on regulatory substance.
                """

                response = self.llm.invoke(summary_prompt)
                summarized_content = extract_content_from_response(response)

                # Create a summary chunk
                summary_chunk = {
                    'content': summarized_content,
                    'text': summarized_content,
                    'metadata': {
                        **metadata_sample,
                        'chunk_type': 'llm_summary',
                        'original_chunks_count': len(chunks),
                        'summary_note': f'Intelligent summary of {len(chunks)} chunks from {doc_id}'
                    }
                }

                logger.info(f"Created LLM summary for large circular {doc_id}")
                return summary_chunk

            else:
                # For moderately long content, create a condensed version
                condensed_chunk = {
                    'content': combined_content[:5000],  # Truncate to manageable size
                    'text': combined_content[:5000],
                    'metadata': {
                        **metadata_sample,
                        'chunk_type': 'condensed',
                        'original_chunks_count': len(chunks),
                        'condensed_note': f'Condensed content from {len(chunks)} chunks'
                    }
                }

                return condensed_chunk

        except Exception as e:
            logger.error(f"Error creating circular summary: {e}")
            # Fallback: return first few chunks
            return {
                'content': '\n\n'.join([chunk.get('content', '') for chunk in chunks[:3]]),
                'text': '\n\n'.join([chunk.get('content', '') for chunk in chunks[:3]]),
                'metadata': chunks[0].get('metadata', {}) if chunks else {}
            }

    def find_latest_circulars_by_topic(self, topic: str, limit: int = 5) -> List[Dict]:
        """Find the latest circulars on a specific topic"""
        try:
            # First try semantic search with document_type filter
            semantic_results = self.embedding_service.search_similar_texts(
                query_text=f"CSSF circular {topic}",
                top_k=limit * 2,  # Get more to filter
                with_scores=True
            )

            # Filter to only include CSSF circulars
            circular_docs = []
            for doc_data in semantic_results:
                if isinstance(doc_data, tuple):
                    doc_content, score = doc_data
                else:
                    doc_content = doc_data
                    score = 0

                metadata = doc_content.get('metadata', {})
                if metadata.get('document_type') == 'CSSF circular':
                    circular_docs.append((doc_content, score))

            # Sort by publication date if available, otherwise by score
            def sort_key(item):
                doc_content, score = item
                metadata = doc_content.get('metadata', {})
                pub_date = metadata.get('publication_date', '')
                # Return tuple (has_date, date, score) for sorting
                return (bool(pub_date), pub_date, score)

            circular_docs.sort(key=sort_key, reverse=True)

            return [doc for doc, score in circular_docs[:limit]]

        except Exception as e:
            logger.error(f"Error finding latest circulars for topic {topic}: {e}")
            return []

    def find_circulars_for_comparison(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Smart method to find two circulars for comparison based on query context
        """
        try:
            # Extract potential topics from query
            topics = self._extract_topics_from_query(query)

            if topics:
                # Find circulars related to these topics
                all_circulars = []
                for topic in topics:
                    circulars = self.find_latest_circulars_by_topic(topic, limit=10)
                    all_circulars.extend(circulars)

                # Remove duplicates based on doc_id
                unique_circulars = {}
                for circular in all_circulars:
                    doc_id = circular.get('metadata', {}).get('doc_id')
                    if doc_id and doc_id not in unique_circulars:
                        unique_circulars[doc_id] = circular

                circular_list = list(unique_circulars.values())

                # If we have at least 2, return the first two
                if len(circular_list) >= 2:
                    return [circular_list[0]], [circular_list[1]]

            # Fallback: semantic search with document_type filter
            semantic_results = self.embedding_service.search_similar_texts(
                query_text=query,
                top_k=20,
                with_scores=True
            )

            # Filter and group by circular
            circular_groups = {}
            for doc_data in semantic_results:
                if isinstance(doc_data, tuple):
                    doc_content, score = doc_data
                else:
                    doc_content = doc_data

                metadata = doc_content.get('metadata', {})
                if metadata.get('document_type') == 'CSSF circular':
                    doc_id = metadata.get('doc_id', metadata.get('title', 'unknown'))
                    if doc_id not in circular_groups:
                        circular_groups[doc_id] = []
                    circular_groups[doc_id].append(doc_content)

            # Return first two groups
            circular_items = list(circular_groups.values())
            if len(circular_items) >= 2:
                return circular_items[0], circular_items[1]

            return [], []

        except Exception as e:
            logger.error(f"Error finding circulars for comparison: {e}")
            return [], []

    def _extract_topics_from_query(self, query: str) -> List[str]:
        """Extract potential topics from comparison query"""
        # Common regulatory topics
        topics = []
        query_lower = query.lower()

        topic_keywords = {
            'aml': ['aml', 'anti-money laundering', 'money laundering'],
            'mifid': ['mifid', 'markets in financial instruments'],
            'ucits': ['ucits', 'undertakings for collective investment'],
            'aifm': ['aifm', 'alternative investment fund managers'],
            'crd': ['crd', 'capital requirements directive'],
            'gdpr': ['gdpr', 'data protection'],
            'outsourcing': ['outsourcing', 'third party'],
            'risk management': ['risk management', 'risk'],
            'governance': ['governance', 'corporate governance'],
            'reporting': ['reporting', 'regulatory reporting']
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _get_title_from_docs(self, docs: List[Dict]) -> str:
        """Extract title from document metadata"""
        for doc in docs:
            if isinstance(doc, dict):
                metadata = doc.get('metadata', {})
                if 'title' in metadata:
                    return metadata['title']
                elif 'title' in doc:
                    return doc['title']
        return None

    def smart_llm_circular_search(self, query: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Use LLM to intelligently find circulars when extraction fails
        """
        try:
            # Ask LLM to help find circulars based on the query
            search_prompt = f"""
            The user wants to compare circulars but we need to find them. Help identify what circulars to search for.

            User Query: "{query}"

            Based on this query, suggest search terms for finding CSSF circulars in a document database.
            Consider that the user might reference circulars in various ways.

            Respond with ONLY this JSON format:
            {{
                "search_terms": ["term1", "term2"],
                "circular_numbers": ["16/635", "12/539"],
                "topics": ["prospectus", "securities"],
                "search_strategy": "specific_numbers|topic_based|semantic_search"
            }}

            Examples:
            For "compare 16/635 and 12/539" → {{"search_terms": ["16/635", "12/539"], "circular_numbers": ["16/635", "12/539"], "search_strategy": "specific_numbers"}}
            For "compare AML circulars" → {{"search_terms": ["AML", "anti-money laundering"], "topics": ["AML"], "search_strategy": "topic_based"}}
            """

            response = self.llm.invoke(search_prompt)
            content = extract_content_from_response(response).strip()
            content = self._clean_json_response(content)

            search_data = json.loads(content)

            circular_numbers = search_data.get('circular_numbers', [])
            search_terms = search_data.get('search_terms', [])
            topics = search_data.get('topics', [])
            strategy = search_data.get('search_strategy', 'semantic_search')

            logger.info(f"LLM search guidance: numbers={circular_numbers}, terms={search_terms}, strategy={strategy}")

            # Execute search based on LLM guidance
            if strategy == "specific_numbers" and len(circular_numbers) >= 2:
                circular1_docs = self.find_circulars_by_identifiers([circular_numbers[0]])
                circular2_docs = self.find_circulars_by_identifiers([circular_numbers[1]])
                return circular1_docs, circular2_docs

            elif strategy == "topic_based" and topics:
                topic_circulars = self.find_latest_circulars_by_topic(topics[0], limit=4)
                if len(topic_circulars) >= 2:
                    return [topic_circulars[0]], [topic_circulars[1]]

            else:
                # Use search terms for semantic search
                all_results = []
                for term in search_terms[:2]:  # Limit to 2 terms
                    results = self.embedding_service.search_similar_texts(
                        query_text=f"CSSF circular {term}",
                        top_k=10,
                        with_scores=True
                    )

                    # Filter for CSSF circulars
                    for doc_data in results:
                        if isinstance(doc_data, tuple):
                            doc_content, score = doc_data
                        else:
                            doc_content = doc_data

                        metadata = doc_content.get('metadata', {})
                        if metadata.get('document_type') == 'CSSF circular':
                            all_results.append(doc_content)

                # Group by circular and return top 2
                if len(all_results) >= 2:
                    circular_groups = {}
                    for doc in all_results:
                        doc_id = doc.get('metadata', {}).get('doc_id', 'unknown')
                        if doc_id not in circular_groups:
                            circular_groups[doc_id] = []
                        circular_groups[doc_id].append(doc)

                    circular_items = list(circular_groups.values())[:2]
                    if len(circular_items) >= 2:
                        return circular_items[0], circular_items[1]

            return [], []

        except Exception as e:
            logger.error(f"Error in smart LLM circular search: {e}")
            return [], []

    def enhanced_find_circulars_by_context(self, params: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """
        Enhanced method to find circulars using LLM-extracted parameters with fallback to smart search
        """
        try:
            circular_identifiers = params.get('circulars', [])
            temporal_context = params.get('temporal_context')
            topic_area = params.get('topic_area')
            confidence = params.get('confidence', 0.5)

            logger.info(
                f"Finding circulars with context: identifiers={circular_identifiers}, temporal={temporal_context}, topic={topic_area}")

            # Strategy 1: Specific circular identifiers from LLM
            if len(circular_identifiers) >= 2:
                logger.info("Using LLM-extracted circular identifiers")
                circular1_docs = self.find_circulars_by_identifiers([circular_identifiers[0]])
                circular2_docs = self.find_circulars_by_identifiers([circular_identifiers[1]])

                if circular1_docs and circular2_docs:
                    return circular1_docs, circular2_docs
                else:
                    # Log which ones weren't found for better debugging
                    not_found = []
                    if not circular1_docs:
                        not_found.append(circular_identifiers[0])
                    if not circular2_docs:
                        not_found.append(circular_identifiers[1])
                    logger.warning(f"Could not find circulars using identifiers: {not_found}")

                    # If we found at least one circular, try to find a related one
                    if circular1_docs and not circular2_docs:
                        logger.info("Found first circular, searching for related circular")
                        # Try to find a related circular
                        related_docs = self._find_related_circular(circular1_docs[0])
                        if related_docs:
                            return circular1_docs, related_docs
                    elif circular2_docs and not circular1_docs:
                        logger.info("Found second circular, searching for related circular")
                        related_docs = self._find_related_circular(circular2_docs[0])
                        if related_docs:
                            return related_docs, circular2_docs

            # Strategy 2: Single identifier + temporal context
            elif len(circular_identifiers) == 1 and temporal_context:
                logger.info("Using single identifier with temporal context")
                specific_docs = self.find_circulars_by_identifiers([circular_identifiers[0]])

                if topic_area:
                    related_docs = self.find_latest_circulars_by_topic(topic_area, limit=5)
                    related_docs = [doc for doc in related_docs
                                    if doc.get('metadata', {}).get('doc_id') !=
                                    (specific_docs[0].get('metadata', {}).get('doc_id') if specific_docs else None)]
                    if related_docs:
                        return specific_docs, [related_docs[0]]

                return specific_docs, []

            # Strategy 3: Topic-based with temporal context
            elif topic_area and temporal_context:
                logger.info("Using topic-based search with temporal context")

                if temporal_context in ['latest_vs_previous', 'new_vs_old']:
                    latest_circulars = self.find_latest_circulars_by_topic(topic_area, limit=5)

                    if len(latest_circulars) >= 2:
                        sorted_circulars = self._sort_circulars_by_date(latest_circulars)
                        return [sorted_circulars[0]], [sorted_circulars[1]]

                    return latest_circulars[:1], latest_circulars[1:2] if len(latest_circulars) > 1 else []

            # Strategy 4: Topic-based search only
            elif topic_area:
                logger.info("Using topic-based search")
                topic_circulars = self.find_latest_circulars_by_topic(topic_area, limit=4)

                if len(topic_circulars) >= 2:
                    return [topic_circulars[0]], [topic_circulars[1]]

                return topic_circulars[:1], topic_circulars[1:2] if len(topic_circulars) > 1 else []

            # Strategy 5: If everything else failed, use smart LLM search
            else:
                logger.info("Using smart LLM-guided search as final fallback")
                result = self.smart_llm_circular_search(f"{topic_area or ''} circular comparison")
                if result:
                    return result

            # Return empty lists if nothing found
            logger.warning("No circulars found with any strategy")
            return [], []

        except Exception as e:
            logger.error(f"Error in enhanced circular finding: {e}")
            return [], []

    def _find_related_circular(self, reference_doc: Dict) -> List[Dict]:
        """Find a circular related to the reference document"""
        try:
            metadata = reference_doc.get('metadata', {})
            title = metadata.get('title', '')

            # Look for related circulars mentioned in the title
            # Example: "Circular CSSF 16/635 (outdated)" might be related to other prospectus circulars
            if 'prospectus' in title.lower():
                related_circulars = self.find_latest_circulars_by_topic('prospectus', limit=3)
                # Return a different circular than the reference
                for circular in related_circulars:
                    if circular.get('metadata', {}).get('doc_id') != metadata.get('doc_id'):
                        return [circular]

            # Try to find any other CSSF circular
            try:
                all_circulars = self.embedding_service.search_similar_texts(
                    query_text="CSSF circular",
                    top_k=10,
                    with_scores=True
                )

                for doc_data in all_circulars:
                    if isinstance(doc_data, tuple):
                        doc_content, score = doc_data
                    else:
                        doc_content = doc_data

                    doc_metadata = doc_content.get('metadata', {})
                    if (doc_metadata.get('document_type') == 'CSSF circular' and
                            doc_metadata.get('doc_id') != metadata.get('doc_id')):
                        return [doc_content]

            except Exception as e:
                logger.error(f"Error finding related circular: {e}")

            return []

        except Exception as e:
            logger.error(f"Error in _find_related_circular: {e}")
            return []

    def _sort_circulars_by_date(self, circulars: List[Dict]) -> List[Dict]:
        """Sort circulars by publication date, newest first"""

        def get_date_key(circular):
            metadata = circular.get('metadata', {})
            pub_date = metadata.get('publication_date', '')
            update_date = metadata.get('update_date', '')

            # Use publication date if available, otherwise update date
            date_str = pub_date or update_date

            # Return the date string for sorting (ISO format sorts correctly)
            return date_str or '0000-00-00'

        try:
            return sorted(circulars, key=get_date_key, reverse=True)
        except Exception as e:
            logger.error(f"Error sorting circulars by date: {e}")
            return circulars

    def analyze_regulatory_changes(self, current_docs: List[Dict], previous_docs: List[Dict],
                                   params: Dict[str, Any]) -> str:
        """
        Dedicated method for analyzing regulatory changes with impact focus
        """

        # Extract content from documents
        current_content = self._extract_content_from_docs(current_docs)
        previous_content = self._extract_content_from_docs(previous_docs)

        # Get titles and metadata
        current_title = self._get_title_from_docs(current_docs) or 'Current Version'
        previous_title = self._get_title_from_docs(previous_docs) or 'Previous Version'

        # Extract dates for timeline analysis
        current_date = self._get_publication_date(current_docs)
        previous_date = self._get_publication_date(previous_docs)

        # Build context for change analysis
        context_info = []

        if params.get('topic_area'):
            context_info.append(f"Regulatory Area: {params['topic_area']}")

        if params.get('temporal_context'):
            context_info.append(f"Change Context: {params['temporal_context']}")

        if current_date and previous_date:
            context_info.append(f"Timeline: {previous_date} → {current_date}")

        context_section = '\n'.join(context_info) if context_info else ""

        change_analysis_prompt = f"""
        You are a regulatory compliance expert specializing in change impact analysis for Luxembourg and European financial regulations.

        Analyze the regulatory changes between these two CSSF circular versions and provide a comprehensive impact assessment.

        {context_section}

        **CURRENT VERSION: {current_title}**
        {current_content[:3000]}

        **PREVIOUS VERSION: {previous_title}**
        {previous_content[:3000]}

        Provide a detailed change analysis covering:

        ## Change Summary
        - Brief overview of the nature and scope of changes
        - Key drivers behind the regulatory updates

        ## Detailed Change Analysis
        ### New Requirements
        - What new obligations have been introduced
        - New compliance standards or procedures
        - Additional reporting or documentation requirements

        ### Modified Requirements  
        - What existing requirements have been changed
        - How procedures or standards have been updated
        - Changes in thresholds, timelines, or criteria

        ### Removed/Relaxed Requirements
        - What requirements have been eliminated
        - Areas where compliance has been simplified
        - Reduced reporting or administrative burdens

        ## Impact Assessment
        ### Immediate Impact (0-6 months)
        - Urgent compliance actions required
        - Immediate operational changes needed
        - Quick wins or easy implementations

        ### Medium-term Impact (6-12 months)
        - Process redesign requirements
        - System updates or implementations
        - Training and capability building needs

        ### Long-term Impact (12+ months)
        - Strategic implications for business operations
        - Competitive advantages or challenges
        - Industry-wide transformation effects

        ## Risk Analysis
        ### Compliance Risks
        - Non-compliance penalties and consequences
        - Areas of regulatory uncertainty
        - Potential interpretation challenges

        ### Operational Risks
        - Implementation complexities
        - Resource requirements and constraints
        - Timeline pressures

        ### Business Risks
        - Market impact and competitive implications
        - Customer or stakeholder effects
        - Financial implications

        ## Implementation Roadmap
        ### Phase 1: Immediate Actions (Next 30 days)
        - Critical compliance steps
        - Stakeholder notifications
        - Initial assessments

        ### Phase 2: Short-term Implementation (30-90 days)
        - Process updates
        - Staff training
        - System modifications

        ### Phase 3: Full Implementation (90+ days)
        - Complete process overhaul
        - Advanced system integrations
        - Continuous monitoring setup

        ## Recommendations
        ### Priority Actions
        - Most critical steps to take immediately
        - High-impact, low-effort improvements
        - Risk mitigation priorities

        ### Resource Allocation
        - Recommended team structure
        - Budget considerations
        - External support requirements

        ### Success Metrics
        - Key performance indicators for implementation
        - Compliance monitoring measures
        - Business impact measurements

        ## Conclusion
        - Overall assessment of change significance
        - Strategic recommendations for leadership
        - Key success factors for implementation

        Focus on actionable insights, specific timelines, and practical implementation guidance.
        Highlight the most critical changes that require immediate attention.
        """

        response = self.llm.invoke(change_analysis_prompt)
        return extract_content_from_response(response)

    def _get_publication_date(self, docs: List[Dict]) -> str:
        """Extract publication date from document metadata"""
        for doc in docs:
            if isinstance(doc, dict):
                metadata = doc.get('metadata', {})
                pub_date = metadata.get('publication_date')
                if pub_date:
                    return pub_date

                # Fallback to update date
                update_date = metadata.get('update_date')
                if update_date:
                    return update_date

        return None

    def compare_circulars_with_context(self, circular1_docs: List[Dict], circular2_docs: List[Dict],
                                       params: Dict[str, Any]) -> str:
        """Compare two sets of circular documents with enhanced context from LLM extraction"""

        # Extract content from documents
        content1 = self._extract_content_from_docs(circular1_docs)
        content2 = self._extract_content_from_docs(circular2_docs)

        # Get titles for reference
        title1 = self._get_title_from_docs(circular1_docs) or 'First Circular'
        title2 = self._get_title_from_docs(circular2_docs) or 'Second Circular'

        # Build context-aware prompt
        context_info = []

        if params.get('temporal_context'):
            temporal_map = {
                'latest_vs_previous': 'comparing the latest version with the previous version',
                'new_vs_old': 'comparing the new circular with the old circular',
                'recent_changes': 'focusing on recent changes and updates'
            }
            context_info.append(f"Context: {temporal_map.get(params['temporal_context'], 'temporal comparison')}")

        if params.get('topic_area'):
            context_info.append(f"Topic Focus: {params['topic_area']} regulations")

        if params.get('specific_focus'):
            context_info.append(f"Specific Focus: {params['specific_focus']}")

        context_section = '\n'.join(context_info) if context_info else ""

        comparison_prompt = f"""
        You are a financial regulatory expert specializing in Luxembourg and European regulations. 
        Compare these two CSSF circulars and provide a comprehensive analysis.

        {context_section}

        **CIRCULAR 1: {title1}**
        {content1[:2500]}

        **CIRCULAR 2: {title2}**
        {content2[:2500]}

        Provide a structured comparison covering:

        ## Executive Summary
        - Brief overview of what each circular covers
        - Key relationship between the two circulars

        ## Key Similarities
        - What regulatory principles remain consistent
        - Common requirements and obligations
        - Unchanged compliance standards

        ## Key Differences
        - New requirements introduced
        - Modified obligations or procedures
        - Removed or relaxed requirements
        - Changes in scope or applicability

        ## Impact Assessment
        - What these changes mean for financial institutions
        - Compliance implications and timeline
        - Operational changes required
        - Risk management considerations

        ## Action Items
        - Immediate steps institutions should take
        - Compliance deadlines to note
        - Documentation or process changes required
        - Training or system updates needed

        ## Conclusion
        - Overall significance of the changes
        - Strategic implications for institutions

        Use clear headings and bullet points for easy reading.
        Focus on actionable insights for compliance teams.
        Cite specific sections or articles when possible.
        """

        response = self.llm.invoke(comparison_prompt)
        return extract_content_from_response(response)

    def _extract_content_from_docs(self, docs: List[Dict]) -> str:
        """Extract and combine content from document list"""
        content_parts = []

        for doc in docs:
            if isinstance(doc, dict):
                if 'text' in doc:
                    content_parts.append(doc['text'])
                elif 'content' in doc:
                    content_parts.append(doc['content'])
                elif 'page_content' in doc:
                    content_parts.append(doc['page_content'])

        return '\n\n'.join(content_parts)

    def _clean_json_response(self, content: str) -> str:
        """Clean up common JSON formatting issues from LLM responses"""
        try:
            # Remove markdown code blocks
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])  # Remove first and last lines

            # Remove any "json" language identifier
            content = content.replace('```json', '').replace('```', '')

            # Strip whitespace
            content = content.strip()

            # Find JSON object boundaries
            start_idx = content.find('{')
            end_idx = content.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx + 1]

            return content

        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            return content


def process_comparison_workflow(query: str, intent: QueryIntent, params: Dict[str, Any],
                                embedding_service, llm) -> Dict[str, Any]:
    """Process comparison-specific workflows with detailed debugging"""

    logger.info(f"🔄 STARTING comparison workflow for intent: {intent.value}")
    logger.info(f"📋 Parameters received: {params}")

    comparator = CircularComparator(embedding_service, llm)

    try:
        # Use string comparison instead of enum comparison
        if intent.value == 'compare_circulars':
            logger.info("🔍 Processing COMPARE_CIRCULARS workflow")

            # Use enhanced circular finding with focus on temporal context
            circular1_docs, circular2_docs = comparator.enhanced_find_circulars_by_context(params)

            logger.info(
                f"📊 Search results: circular1={len(circular1_docs) if circular1_docs else 0} docs, circular2={len(circular2_docs) if circular2_docs else 0} docs")

            if not circular1_docs or not circular2_docs:
                # More helpful error message based on what was found
                if params.get('topic_area'):
                    topic_msg = f" related to {params['topic_area']}"
                else:
                    topic_msg = ""

                if params.get('temporal_context'):
                    temporal_msg = f" with temporal context '{params['temporal_context']}'"
                else:
                    temporal_msg = ""

                logger.info(f"⚠️ Insufficient circulars found for comparison")
                return {
                    'answer': f"I couldn't find enough distinct CSSF circulars to compare{topic_msg}{temporal_msg}. I found {len(circular1_docs) if circular1_docs else 0} and {len(circular2_docs) if circular2_docs else 0} circular groups. Please be more specific about which circulars you'd like me to compare (e.g., 'Compare Circular 23/123 with Circular 24/456', or 'Compare the latest AML circular with the previous AML circular').",
                    'sources': [],
                    'workflow': 'comparison_insufficient_circulars',
                    'intent': intent.value,
                    'extracted_params': params
                }

            # Get titles for logging
            title1 = comparator._get_title_from_docs(circular1_docs) or "First Circular"
            title2 = comparator._get_title_from_docs(circular2_docs) or "Second Circular"
            logger.info(f"✅ Successfully found circulars to compare: '{title1}' vs '{title2}'")

            # Enhanced comparison with context
            logger.info("🔄 Generating comparison analysis...")
            comparison_result = comparator.compare_circulars_with_context(
                circular1_docs,
                circular2_docs,
                params
            )

            logger.info("✅ Comparison workflow completed successfully")
            return {
                'answer': comparison_result,
                'sources': [],
                'workflow': 'comparison_completed',
                'intent': intent.value,
                'circular1_title': title1,
                'circular2_title': title2,
                'extracted_params': params
            }

        elif intent.value == 'find_differences':
            logger.info("🔍 Processing FIND_DIFFERENCES workflow - routing to comparison")
            # Similar to comparison but focused on differences
            return process_comparison_workflow(query, intent, params, embedding_service, llm)

        else:
            logger.warning(f"⚠️ Unsupported intent for comparison workflow: {intent.value}")
            # Fallback to general query processing
            return None

    except Exception as e:
        logger.error(f"❌ Error in comparison workflow: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'answer': f"I encountered an error while processing your comparison request: {str(e)}",
            'sources': [],
            'workflow': 'comparison_error',
            'intent': intent.value
        }


# =====================================================
# EXISTING UTILITY FUNCTIONS (UNCHANGED)
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
# ENHANCED QUERY PROCESSING WITH AGENTIC AI
# =====================================================

def enhanced_process_query_action(query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Enhanced query processing with intent detection and specialized workflows"""

    logger.info(f"🎯 ENHANCED_PROCESS_QUERY_ACTION called with query: '{query}'")

    try:
        # Step 1: Initialize intent detector
        intent_detector = IntentDetector(llm)
        logger.info("🔧 Intent detector initialized")

        # Step 2: Detect intent
        intent, params = intent_detector.detect_intent(query)
        logger.info(f"🎯 Detected intent: {intent.value}, params: {params}")

        # Step 3: Route to appropriate workflow - FIXED: Use string comparison
        logger.info(f"🔍 Checking intent matching: intent={intent}, intent.value={intent.value}")
        logger.info(f"🔍 Using string comparison for reliability")

        if intent.value in ['compare_circulars', 'find_differences']:
            # Check if we have circular identifiers (from LLM or backup extraction)
            circular_identifiers = params.get('circulars', [])

            logger.info(
                f"🔄 INTENT MATCHED! Processing {intent.value} with circular identifiers: {circular_identifiers}")

            # ALWAYS try the comparison workflow for comparison intents - don't check conditions
            try:
                logger.info("🔄 About to call process_comparison_workflow...")
                result = process_comparison_workflow(query, intent, params, embedding_service, llm)
                logger.info(f"🔄 process_comparison_workflow returned: {type(result)} - {bool(result)}")

                if result:  # If comparison workflow handled it
                    logger.info("✅ Comparison workflow completed successfully, returning result")
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*',
                        },
                        'body': json.dumps({
                            **result,
                            'success': True
                        })
                    }
                else:
                    logger.warning("⚠️ Comparison workflow returned None/empty result")
            except Exception as comp_error:
                logger.error(f"❌ Error in comparison workflow: {comp_error}")
                import traceback
                logger.error(f"❌ Comparison workflow traceback: {traceback.format_exc()}")

        elif intent.value == 'analyze_changes':
            logger.info("🔄 Processing ANALYZE_CHANGES workflow")
            # Dedicated change analysis workflow
            logger.info("Processing change analysis workflow")

            comparator = CircularComparator(embedding_service, llm)

            # Use enhanced circular finding with focus on temporal context
            circular1_docs, circular2_docs = comparator.enhanced_find_circulars_by_context(params)

            if not circular1_docs or not circular2_docs:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                    },
                    'body': json.dumps({
                        'answer': f"I need to find two versions of a circular to analyze changes. Please specify which circulars you'd like me to analyze (e.g., 'Analyze changes in Circular 23/123 compared to the previous version', or 'What changed in the latest AML circular?').",
                        'sources': [],
                        'workflow': 'change_analysis_insufficient_data',
                        'intent': intent.value,
                        'extracted_params': params,
                        'success': True
                    })
                }

            # Get titles for logging
            title1 = comparator._get_title_from_docs(circular1_docs) or "Current Version"
            title2 = comparator._get_title_from_docs(circular2_docs) or "Previous Version"
            logger.info(f"Analyzing changes: {title1} vs {title2}")

            # Dedicated change analysis
            change_analysis = comparator.analyze_regulatory_changes(
                circular1_docs,
                circular2_docs,
                params
            )

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                },
                'body': json.dumps({
                    'answer': change_analysis,
                    'sources': [],
                    'workflow': 'change_analysis_completed',
                    'intent': intent.value,
                    'current_version': title1,
                    'previous_version': title2,
                    'extracted_params': params,
                    'success': True
                })
            }
        else:
            logger.info(f"🔄 Intent {intent.value} not matched for special workflows")

        # Step 4: Fallback to general query processing
        logger.info("🔄 Using fallback general query processing")
        return process_query_action(query, history)

    except Exception as e:
        logger.error(f"❌ Error in enhanced query processing: {e}")
        import traceback
        logger.error(f"❌ Enhanced query processing traceback: {traceback.format_exc()}")
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
# EXISTING FUNCTIONS (UNCHANGED)
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
# ENHANCED LAMBDA HANDLER
# =====================================================

def lambda_handler(event, context):
    """Enhanced AWS Lambda handler with agentic AI capabilities"""

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
            # FIXED: Make sure we're calling the enhanced function
            logger.info(f"🚀 Lambda handler calling enhanced_process_query_action with query: '{query}'")
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

    # Set fake env vars (mimic Lambda environment)
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'embedding-endpoint'
    os.environ['MILVUS_HOST'] = '34.241.177.15'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['MILVUS_COLLECTION'] = 'cssf_documents_final_final'
    os.environ['BEDROCK_REGION'] = 'us-east-1'
    os.environ['IRELAND_REGION'] = 'eu-west-1'

    # Test parameter extraction first
    print("Testing parameter extraction...")


    # Test the backup extraction directly
    class MockLLM:
        def invoke(self, prompt):
            return '{"circular_identifiers": [], "confidence": 0.9}'  # Simulate LLM failure


    detector = IntentDetector(MockLLM())
    test_query = "compare circular 16/635 and circular 12/539"  # Use existing circulars
    params = detector._backup_circular_extraction(test_query)
    print(f"Backup extraction result: {params}")

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