# streamlit_app.py
"""
Streamlit app to query an LLM via AWS Lambda.

This app allows users to enter a query and receive a response from an MCP Server.
"""

import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv
from strands import Agent, tool
from mcp.client.streamable_http import streamablehttp_client
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
# Load environmental variables from a .env file
load_dotenv("env.txt")

QDRANT_API = os.getenv("QDRANT_APIKEY")
QDRANT_URL = os.getenv("QDRANT_URL")
FIRECRAWL_API = os.getenv("FIRECRAWL_APIKEY")
COLLECTION_NAME = "arxiv-collection"

bedrock_model = BedrockModel(
  model_id="amazon.nova-lite-v1:0", 
  region_name="us-east-1",
  temperature=0.3,
  streaming=True, # Enable/disable streaming
)

# Set up the Streamlit app
st.title('Serverless MCP Server')
st.write("Enter your query below and get the response.")

# Input for the query
query = st.text_input("Query:")

# Input for the collection name (optional)
api_gateway_url = <API-GATEWAY-URL>
headers={'collection_name': COLLECTION_NAME, 'qdrant_url': QDRANT_URL}
streamable_http_mcp_client = MCPClient(lambda: streamablehttp_client(api_gateway_url,headers=headers))


# Button to submit the query
if st.button("Submit"):
    if query:
        with st.spinner('Connecting to MCP Server..'):
            with streamable_http_mcp_client:
                tools = streamable_http_mcp_client.list_tools_sync()
                agent = Agent(
                model=bedrock_model,
                system_prompt="You are a smart AI assistant, who responds to user queries",
                tools=tools)
                # Call the agent and return its response
                response = agent(query)
            if response:
                st.write("Response:")
                # st.json(response)
                st.write(str(response))
            else:
                st.error("Something went wrong with MCP Server!")
    else:
        st.error("Please enter a query.")