import logging

from langchain_community.tools import DuckDuckGoSearchRun
from livekit.agents import RunContext, function_tool


@function_tool
async def search_web(context: RunContext, query: str):
    """use this tool to search the web for information related to the user's query

    Args:
        query: The search query provided by the user.

    """
    try:
        result = DuckDuckGoSearchRun().run(tool_input=query)
        logging.info(f"web search result for query '{query}': {result}")
        return result
    except Exception as e:
        logging.error(
            f"Error occurred while searching the web for query '{query}': {e}"
        )
        raise
