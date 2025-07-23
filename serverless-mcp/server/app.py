import logging
import pyjokes
import os
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

load_dotenv("env.txt")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

QDRANT_API = os.getenv("QDRANT_APIKEY")
bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_client)

mcp_server = MCPLambdaHandler(name="mcp-lambda-server", version="1.0.0")


@mcp_server.tool()
def tell_me_jokes(query: str) -> str:
    """Searches the web for the given query and returns the results."""
    # In a real scenario, this would integrate with a web search API
    return pyjokes.get_joke()


@mcp_server.tool()
def get_current_time() -> str:
    """Returns the current date and time."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp_server.tool()
def rag_retrieve_and_generate(query:str, collection_name:str):

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


logger.info("Lambda handler has started!")


def lambda_handler(event, context):
    global collection_name, QDRANT_URL
    headers = event.get("headers", {})
    collection_name = headers.get("collection_name")
    QDRANT_URL = headers.get("qdrant_url")
    result = mcp_server.handle_request(event, context)
    logger.info("Returning responses from mcp server")
    return result