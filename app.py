import json
import os
import sys
from typing import List, Dict, Any
from langchain_aws import BedrockLLM
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from embedding_provider.embedding_provider import EmbeddingService

def lambda_handler(event, context):
    try:

        body = json.loads(event.get('body', '{}'))
        query = body.get('query', '')
        action = body.get('action', 'query')

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
        
        # Initialize Bedrock LLM (Stockholm region)
        llm = BedrockLLM(
            model_id="meta.llama3-70b-instruct-v1:0",  # Adjust model as needed
            region_name=os.environ['BEDROCK_REGION'],
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        )

        if action == 'query':
            # Use your embedding service for similarity search
            similar_docs = embedding_service.search_similar_texts(
                query_text=query,
                top_k=20,  # retrieve more to let LLM rank
                with_scores=True
            )

            # Convert to LangChain documents
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

            # Ask LLM to rank retrieved documents by relevance
            ranking_prompt = f"""Rank the following document snippets by their relevance to the question.
        Each snippet is labeled with a number. Return a JSON list of the top 10 most relevant snippet numbers in order of relevance.

        Question: {query}

        Snippets:
        """ + "\n".join([f"{i + 1}. {doc.page_content[:1000].replace('\n', ' ')}" for i, doc in enumerate(documents)]) + """

        Return only a JSON list like this: [3, 1, 5, ...]
        """

            ranking_response = llm.invoke(ranking_prompt)
            print("------------------------------------------------------------------------------")
            print("Ranking response:", ranking_response)
            print("------------------------------------------------------------------------------")

            try:
                ranked_indices = json.loads(ranking_response)
            except Exception as e:
                print("Failed to parse ranking response:", e)
                ranked_indices = list(range(min(10, len(documents))))

            # Select top-ranked documents
            top_docs = [documents[i - 1] for i in ranked_indices if 1 <= i <= len(documents)][:10]

            # Create final context
            context = "\n\n".join([doc.page_content for doc in top_docs])

            # Compose final prompt to answer question
            final_prompt = f"""Based on the following context, please answer the question.

        Context:
        {context}

        Question: {query}

        Answer:"""

            print("Final PROMPT:", final_prompt)
            print("------------------------------------------------------------------------------")

            response = llm.invoke(final_prompt)

            # Return answer with sources
            #sources = [{
                #'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                #'metadata': doc.metadata
            #} for doc in top_docs]

            top_docs = [documents[i - 1] for i in ranked_indices if 1 <= i <= len(documents)][:10]
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
                    'answer': response,
                    'sources': unique_urls,
                    'success': True
                })
            }
        elif action == 'add_documents':
            documents_data = body.get('documents', [])
            
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
        
        elif action == 'search_similar':
            # Direct similarity search without LLM
            query_text = body.get('query', '')
            top_k = body.get('top_k', 10)
            with_scores = body.get('with_scores', True)
            
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

    except Exception as e:
        import traceback

        try:
            exc_type, exc_value, exc_tb = sys.exc_info()
            if exc_type is not None:
                error_details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            else:
                error_details = str(e)

            print("🛑 Error in RAG handler:\n" + error_details)
        except Exception as log_err:
            print("Failed to log exception:")
            print(f"Original error: {e}")
            print(f"Logging error: {log_err}")

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
    import json

    # Set fake env vars (mimic Lambda environment)
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'embedding-endpoint'
    os.environ['MILVUS_HOST'] = '34.241.177.15'
    os.environ['MILVUS_PORT'] = '19530'
    os.environ['MILVUS_COLLECTION'] = 'cssf_documents'
    os.environ['BEDROCK_REGION'] = 'us-east-1'
    os.environ['IRELAND_REGION'] = 'eu-west-1'

    with open("test_event.json") as f:
        event = json.load(f)

    result = lambda_handler(event, None)
    print(json.dumps(result, indent=2))        