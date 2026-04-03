# MyEvolvingAgent 🤖

> 我的个人 Agent，持续进化，成为最强智能体。

A **highly integrated**, continuously-evolving personal AI agent powered by [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph) and OpenAI — drawing nutrients from the best open-source libraries on GitHub to build a super-professional, specialised assistant.

---

## ✨ Features

| Capability | Description |
|---|---|
| 💬 **Conversational AI** | Multi-turn dialogue with pluggable memory backends (buffer / summary / vector) |
| 🔍 **Web Search** | Real-time DuckDuckGo search — no API key needed |
| 🐙 **GitHub Integration** | Search repos, read files, find code examples directly from GitHub |
| 💻 **Code Analysis** | Analyse Python code structure, count lines, extract docstrings |
| 🧩 **Extensible Tools** | Add/remove tools at runtime — the agent *evolves* without restarting |
| 💾 **Memory Persistence** | Save & restore conversation history as JSON |
| 🎨 **Rich CLI** | Beautiful interactive REPL powered by [Rich](https://github.com/Textualize/rich) and [Typer](https://github.com/tiangolo/typer) |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/hana2262/MyEvolvingAgent.git
cd MyEvolvingAgent

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package with all dependencies
pip install -e ".[dev]"
```

### 2. Configure

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# .env
OPENAI_API_KEY=sk-...          # Required for LLM responses
OPENAI_MODEL=gpt-4o-mini       # Optional – default: gpt-4o-mini
GITHUB_TOKEN=ghp_...           # Optional – raises GitHub API rate limit
MEMORY_TYPE=buffer             # buffer | summary | vector
AGENT_VERBOSE=false            # Set true to see tool call traces
```

### 3. Run

```bash
# Interactive chat REPL
agent chat

# Single question
agent chat -q "What are the top Python LLM frameworks on GitHub?"

# List available tools
agent tools

# Show version
agent version

# Or run directly with Python
python main.py chat
```

---

## 🗂️ Project Structure

```
MyEvolvingAgent/
├── agent/
│   ├── __init__.py          # Public API
│   ├── cli.py               # Rich CLI (Typer)
│   ├── config.py            # Pydantic-settings config
│   ├── core.py              # EvolvingAgent orchestrator
│   ├── memory.py            # Memory backends (buffer/summary/vector)
│   ├── prompts/
│   │   └── __init__.py      # System prompts & templates
│   └── tools/
│       ├── __init__.py
│       ├── code_tool.py     # Python code analysis
│       ├── github_tool.py   # GitHub API integration
│       └── search.py        # DuckDuckGo web search
├── tests/
│   ├── test_config.py
│   ├── test_core.py
│   ├── test_memory.py
│   └── test_tools.py
├── main.py                  # python main.py entry point
├── pyproject.toml
└── .env.example
```

---

## 🧠 Memory Backends

| Backend | Description | Extras needed |
|---|---|---|
| `buffer` | Sliding-window message buffer (default) | — |
| `summary` | Older turns are summarised by the LLM | — |
| `vector` | Semantic retrieval via ChromaDB embeddings | `pip install -e ".[vector]"` |

Set via `MEMORY_TYPE` env var or `AgentConfig(memory_type="summary")`.

---

## 🔧 Extending the Agent (Evolution)

Add any LangChain `BaseTool` at runtime:

```python
from agent import EvolvingAgent
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """A brand-new capability."""
    return f"Custom result for: {query}"

agent = EvolvingAgent()
agent.add_tool(my_custom_tool)
print(agent.tool_names)  # includes 'my_custom_tool'
```

Remove a tool:

```python
agent.remove_tool("web_search")
```

---

## 🧪 Testing

```bash
pytest                    # run all tests
pytest tests/test_memory.py  # specific module
pytest --cov=agent        # with coverage
```

---

## 🗺️ Roadmap

- [x] Core agent with tool-calling
- [x] Web search (DuckDuckGo)
- [x] GitHub integration
- [x] Code analysis tools
- [x] Memory persistence
- [x] Rich CLI
- [ ] LangGraph multi-agent workflow
- [ ] Voice input/output
- [ ] Automatic tool discovery from GitHub
- [ ] Plugin system for community tools
- [ ] Web UI (Streamlit / Gradio)
- [ ] Scheduled tasks & proactive reminders

---

## 📜 License

[MIT](LICENSE)
