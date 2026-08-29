from langchain_community.tools.tavily_search import TavilySearchResults
from src.core.config import settings

def get_web_search_tool():
    if not settings.TAVILY_API_KEY:
        from langchain_community.tools import DuckDuckGoSearchRun
        print("Warning: TAVILY_API_KEY not set. Falling back to DuckDuckGo.")
        return DuckDuckGoSearchRun()
    return TavilySearchResults(k=3)
