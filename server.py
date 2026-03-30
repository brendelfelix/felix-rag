"""
Felix RAG - MCP Server (HTTP/SSE transport for Railway)
Exposes search_felix_rag as a Claude tool via a hosted web service.
"""
import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "felix-rag")
EMBED_MODEL      = "llama-text-embed-v2"
TOP_K            = 5
MIN_SCORE        = 0.35


def _search(query: str) -> str:
    from pinecone import Pinecone
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    result = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )
    embedding = result[0]["values"]
    matches = index.query(vector=embedding, top_k=TOP_K, include_metadata=True)["matches"]
    hits = [m for m in matches if m["score"] >= MIN_SCORE]
    if not hits:
        return "No relevant context found in Felix's knowledge base."
    lines = [f"FELIX RAG CONTEXT ({len(hits)} results for: {query})"]
    for i, m in enumerate(hits, 1):
        meta = m["metadata"]
        lines.append(f"[{i}] {meta.get('title', 'Unknown')} (relevance: {m['score']:.0%})")
        lines.append(meta.get("text", ""))
        lines.append("")
    return "\n".join(lines)


from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import Response
import uvicorn

app = Server("felix-rag")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(
        name="search_felix_rag",
        description="Search Felix Brendel personal knowledge base. Use for context on The Lab, Caiya, Tetrad, strategy, goals, clients, decisions.",
        inputSchema={"type":"object","properties":{"query":{"type":"string","description":"Natural language question to search"}},"required":["query"]}
    )]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "search_felix_rag":
        raise ValueError(f"Unknown tool: {name}")
    query = arguments.get("query", "").strip()
    return [types.TextContent(type="text", text=_search(query) if query else "Empty query.")]

def create_app(mcp_server: Server) -> Starlette:
    sse = SseServerTransport("/messages/")
    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
        return Response()
    return Starlette(routes=[Route("/sse", endpoint=handle_sse), Mount("/messages/", app=sse.handle_post_message)])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(create_app(app), host="0.0.0.0", port=port)
