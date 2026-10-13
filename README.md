# Retrieval-Augmented Generation with LangChain and ChromaDB

Run the workflow either in [RAG_Ollama_Beginners.ipynb](RAG_Ollama_Beginners.ipynb) or with the Python scripts below. The notebook remains self-contained; the Python package uses the same settings and workflow.

The notebook contains the complete workflow in eight short steps: dependencies and configuration, document ingestion, text chunking, embedding model initialization, embedding generation and vector indexing, semantic retrieval, prompt engineering and RAG chain construction, and context assembly and response generation.

## Project structure

```text
CITS_tutorial/
├── RAG_Ollama_Beginners.ipynb
├── data/                     # Input PDFs
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── api.py            # FastAPI endpoint 
│       ├── config.py         
│       ├── loader.py         # PDF pages and source metadata
│       ├── chunker.py        
│       ├── embedder.py       # Cached CPU embedding model
│       ├── retriever.py      # Chroma indexing and semantic retrieval
│       ├── generator.py      
│       └── pipeline.py       # Complete RAG chain
├── scripts/
│   ├── setup_models.py       # Download embedding and Ollama models
│   └── ask.py                # Command-line entry point
├── requirements.txt
├── .env.example              
├── .gitignore
└── README.md
```

`chroma_db/` stores the generated persistent index.

## Set up the Python environment

Choose **one** of the following options. Run the commands from this project
folder. Both options install the script and notebook dependencies from
`requirements.txt`.

### Option 1: Conda environment (`rag`)

With Conda installed, create and activate the tutorial environment:

```bash
conda create -n rag python=3.12 pip
conda activate rag
python -m pip install -r requirements.txt
```

If `rag` already exists, skip the creation command and run the activation and
installation commands. In each new terminal, run `conda activate rag` before
using the project.

### Option 2: Python virtual environment and pip

With Python 3.12 installed, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt, use `.venv\Scripts\activate.bat` instead.
Then install the dependencies in the active environment:

```bash
python -m pip install -r requirements.txt
```


## Optional configuration with `.env`

The Python scripts work with the defaults in `src/rag/config.py`. To customize
them, copy `.env.example` to `.env` in the project folder:

```bash
cp .env.example .env
```

On Windows PowerShell or Command Prompt, use `copy .env.example .env`.
Edit `.env` as needed:

```dotenv
OLLAMA_MODEL=qwen3:8b
OLLAMA_URL=http://localhost:11434
RETRIEVAL_K=4
```

`OLLAMA_MODEL` selects the chat model, `OLLAMA_URL` selects the Ollama server,
and `RETRIEVAL_K` sets the maximum number of retrieved passages (a positive
integer). After changing the model or server, rerun `python scripts/setup_models.py`
to pull the model onto that server. Restart any existing Python session to pick
up configuration changes.

The scripts load `.env` from the project folder even when launched elsewhere.
Existing shell environment variables take precedence over `.env`; omitted
values use the defaults. `.env` is ignored by Git, while `.env.example` is
the shared template. The notebook keeps its own settings and does not load `.env`.
The `python-dotenv` dependency is included in `requirements.txt`; if you already
set up the project, rerun `python -m pip install -r requirements.txt` in your
active environment.

## Run the Python scripts

Keep the environment chosen above active for the following commands.

Install [Ollama](https://ollama.com/download), then start `ollama serve` in a
separate terminal if Ollama is not already running. Download the models once:

```bash
python scripts/setup_models.py
```

Model setup needs an internet connection and enough disk space for both models.
The embedding model uses the exact revision pinned in the notebook. Subsequent
questions load that model from the local cache and use the local Ollama server.

Put PDFs in `data/`, then ask a question:

```bash
python scripts/ask.py "Do qualified students receive full scholarships?"
```

## Serve RAG through FastAPI

Install the updated dependencies, then start the API from the project folder:

```bash
python -m uvicorn rag.api:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

To allow other devices on your network to connect, use `--host 0.0.0.0`
instead. They can open `http://<your-computer-LAN-IP>:8000/docs` and send
requests to `http://<your-computer-LAN-IP>:8000/ask`.


```bash
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Do qualified students receive full scholarships?"}'
```

`POST /ask` returns `{"answer": "..."}`; the answer includes the existing
source citations. Questions are independent and no chat history is stored.
Blank or missing questions return HTTP 422. Local inference requests are
processed one at a time. Use one worker to keep one loaded embedding model;
each additional worker would load its own copy.

## Reuse document embeddings

Both the API and `scripts/ask.py` now compare PDF content hashes with an index
manifest stored in `chroma_db/`. Unchanged PDFs reuse their saved embeddings.
Only new or modified PDFs are embedded; old passages from modified or deleted
PDFs are removed.

## Use RAG as an MCP tool

`src/mcp_server.py` exposes an `ask_rag` tool using the official MCP Python
SDK 2.x. An MCP-compatible assistant can discover and call this tool with
`{"question": "Do qualified students receive full scholarships?"}`. It returns
answer text and source citations from the same RAG pipeline.

### HTTP connection

Start the MCP server from the project folder:

```bash
python src/mcp_server.py --transport streamable-http --port 8001
```

In your MCP client, add a server with transport **Streamable HTTP** and URL
`http://127.0.0.1:8001/mcp`. Once connected, ask the assistant to use
`ask_rag` to answer your document question.

For access from other devices on your network:

```bash
python src/mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8001
```

Those clients use `http://<your-computer-LAN-IP>:8001/mcp`.


### MCP connection quick reference

| Setting                        | Value                           |
| ------------------------------ | ------------------------------- |
| Server source                  | `src/mcp_server.py`             |
| Transport                      | Streamable HTTP                 |
| On this computer               | `http://127.0.0.1:8001/mcp`     |
| From another device on the LAN | `http://192.168.1.123:8001/mcp` |
| Tool name                      | `ask_rag`                       |
| Required argument              | `question` (a non-empty string) |

The LAN URL uses this computer's current address. If its address changes,
replace `192.168.1.123` with the new address shown by `hostname -I`.
Keep the MCP server process running while clients use it.

To start it for network access from the project folder:

```bash
conda activate ahim
python src/mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8001
```

Add the URL above in your MCP client's server settings, select Streamable
HTTP, and connect. Ask the assistant: **Use ask_rag to answer:

```json
{
  "question": "Do qualified students receive full scholarships?"
}
```
