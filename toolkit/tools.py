import os

from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from data_models.models import RagToolSchema
from langchain_pinecone import PineconeVectorStore
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

api_wrapper = PolygonAPIWrapper()
model_loader = ModelLoader()
config = load_config()


# ---------------------------------------------------------
# RAG / Pinecone Tool
# ---------------------------------------------------------

@tool(args_schema=RagToolSchema)
def retriever_tool(question):
    """ALWAYS use this tool first when the user asks to summarize, analyze,
    explain, or find information from uploaded documents, files, reports,
    or the knowledge base. This tool searches the internal vector database
    of uploaded documents and returns relevant content."""

    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is missing from .env"
        )

    pc = Pinecone(api_key=pinecone_api_key)

    vector_store = PineconeVectorStore(
        index=pc.Index(
            config["vector_db"]["index_name"]
        ),
        embedding=model_loader.load_embeddings()
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": config["retriever"]["top_k"],
            "score_threshold": config["retriever"]["score_threshold"]
        },
    )

    retriever_result = retriever.invoke(question)

    # IMPORTANT:
    # Don't send huge document chunks to Groq.
    max_chars_per_doc = 3000

    results = []

    for doc in retriever_result:
        content = doc.page_content[:max_chars_per_doc]
        results.append(content)

    return "\n\n---\n\n".join(results)


# ---------------------------------------------------------
# Tavily Web Search Tool
# ---------------------------------------------------------

tavilytool = TavilySearch(
    max_results=config["tools"]["tavily"]["max_results"],
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    description=(
        "Search the web for latest news, market trends, "
        "or any real-time financial information."
    )
)


# ---------------------------------------------------------
# Polygon Financial Tool
# ---------------------------------------------------------

polygon_financials = PolygonFinancials(
    api_wrapper=api_wrapper,
    description=(
        "Get financial statements and data for publicly traded "
        "companies using their stock ticker symbol."
    )
)


@tool
def financials_tool(ticker: str):
    """
    Get financial information for a publicly traded company.

    Returns a limited amount of Polygon data so that the result
    does not exceed the Groq free-tier token limit.
    """

    result = polygon_financials.invoke({
        "ticker": ticker
    })

    # Convert Polygon result to text
    result_text = str(result)

    # IMPORTANT:
    # Polygon can return a very large response.
    # Limit it before sending it to the LLM.
    max_chars = 6000

    if len(result_text) > max_chars:
        result_text = (
            result_text[:max_chars]
            + "\n\n[Financial data truncated to fit the LLM context limit.]"
        )

    return result_text