import logging
import pyjokes
import os, json, requests
import requests
import os
import boto3
from awslabs.mcp_lambda_handler import MCPLambdaHandler
from dotenv import load_dotenv
from uuid import uuid4
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import Qdrant
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain.embeddings import BedrockEmbeddings
from typing import List, Optional

load_dotenv("env.txt")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

QDRANT_API = os.getenv("QDRANT_APIKEY")
FIRECRAWL_API = os.getenv("FIRECRAWL_APIKEY")

bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_client)

mcp_server = MCPLambdaHandler(name="mcp-lambda-server", version="1.0.0")


@mcp_server.tool()
def rag_retrieve_and_generate(query:str):
    """
    Retrieve context from Qdrant and generate an answer using Amazon Nova Pro.
    Use this tool only when the query is about Deep Learning Concepts.

    Args:
    ----
    query (str): The question to retrieve context for.
    collection_name (str): The name of the Qdrant collection to search in.

    Returns:
    -------
    str: The generated answer based on retrieved context.

    """

    # Initialize vector store
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API, port=6333, grpc_port=6333
    )
    vectorstore = Qdrant(client=qdrant_client,
                        collection_name=collection_name,
                        embeddings=bedrock_embeddings,
                        vector_name="content")

    # Define the prompt template
    template = """

    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    


    Question: {question}
    Context: {context}

    Answer:

    """

    # Initialize retriever
    retriever = vectorstore.as_retriever()

    # Create prompt using the template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the LLM (aws-nova-pro)
    llm_novapro = ChatBedrockConverse(
        client=bedrock_client,
        model="amazon.nova-pro-v1:0",
        temperature=0.1,
    )


    # Create a retrieval QA chain
    qa_d35 = RetrievalQA.from_chain_type(llm=llm_novapro,
                                         chain_type="stuff",
                                         chain_type_kwargs = {"prompt": prompt},
                                         retriever=retriever)

    # Invoke the chain with the query to get the result
    result = qa_d35.invoke({"query": query})["result"]
    return result


@mcp_server.tool()
def firecrawl_web_search_tool(query: str) -> List[str]:
    """
    Search for information on a given topic using Firecrawl.
    Use this tool when the user asks a specific question not related to Deep Learning concepts.

    Args:
        query (str): The user query to search for information.

    Returns:
       str: Content for the use query.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    url = "https://api.firecrawl.dev/v1/search"
    api_key =FIRECRAWL_API

    if not api_key:
        return ["Error: FIRECRAWL_API_KEY environment variable is not set."]

    payload = {"query": query, "timeout": 60000, "limit":1, "scrapeOptions": {
      "formats": ["markdown"]
    }}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Assuming the API returns JSON with a key like "data" or "results"
        # Adjust .get("data", ...) if the key is different
        return response.json().get("data", ["No results found from web search."])[0]['markdown']
    except requests.exceptions.RequestException as e:
        return [f"Error connecting to Firecrawl API: {e}"]
    
    
logger.info("Lambda handler has started!")


def lambda_handler(event, context):
    global collection_name, QDRANT_URL
    headers = event.get("headers", {})
    collection_name = headers.get("collection_name")
    QDRANT_URL = headers.get("qdrant_url")
    result = mcp_server.handle_request(event, context)
    logger.info("Returning responses from mcp server")
    return result