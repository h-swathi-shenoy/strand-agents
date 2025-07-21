from awslabs.mcp_lambda_handler import MCPLambdaHandler
import logging
import pyjokes

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


logger.info("Lambda handler has started!")


def lambda_handler(event, context):
    result = mcp_server.handle_request(event, context)
    logger.info("Returning responses from mcp server")
    return result