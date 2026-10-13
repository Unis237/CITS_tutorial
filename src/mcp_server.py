"""Expose the PDF RAG pipeline as an MCP tool over stdio or HTTP."""
import argparse
from contextlib import asynccontextmanager, redirect_stdout
import sys
from threading import Lock

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


def create_server():
    chain = None
    query_lock = Lock()

    @asynccontextmanager
    async def lifespan(server):
        nonlocal chain
        # stdout belongs exclusively to the MCP protocol in stdio mode.
        with redirect_stdout(sys.stderr):
            from rag.pipeline import build_pipeline
            chain = build_pipeline()
        try:
            yield
        finally:
            chain = None

    server = MCPServer("PDF RAG", lifespan=lifespan)

    @server.tool()
    async def ask_rag(question: str) -> str:
        """Answer a question using the local PDFs, returning the answer and source citations.

        Use for questions about the indexed documents. Questions are independent;
        the tool does not retain conversation history.
        """
        question = question.strip()
        if not question:
            raise ToolError("Question must not be empty.")

        def answer():
            with query_lock:
                if chain is None:
                    raise RuntimeError("RAG pipeline is not ready.")
                return chain.invoke(question)

        return await anyio.to_thread.run_sync(answer)

    return server


mcp = create_server()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001).")
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
