# Natural Language Backtesting Service

AI-powered backtesting service that converts natural language trading strategies into executable Python code.

## Features

- **Natural Language Strategy Input**: Describe your trading strategy in plain language
- **AI Code Generation**: Automatically generates backtesting code using LLM
- **Interactive Dashboard**: Visualize backtest results with Plotly charts
- **Multiple Data Providers**: Supports Supabase, YFinance, and mock data
- **Docker Execution**: Secure code execution in isolated Docker containers

## Requirements

- Python 3.13+
- Docker (optional, for secure code execution)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/George-Polya/backtest-simulation.git
cd backtest-simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. (Optional) Build Docker image for secure execution:
```bash
cd docker
docker build -t backtest-runner:latest .
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
# LLM Provider: openrouter 
llm:
  provider: "openrouter"
  model: "openai/gpt-5.2"

# Data Provider: supabase | yfinance | mock
data:
  provider: "supabase"

# Execution: docker | local
execution:
  provider: "docker"
```

## Running

Start the dashboard:
```bash
uvicorn -m app.main:app --port 8050
```

The dashboard will be available at `http://localhost:8050`

## Environment Variables (`.env.example`)

Copy `.env.example` to create your `.env` file and fill in the required API keys and configuration values.

```bash
cp .env.example .env
```

> ⚠️ **Warning**: Never commit `.env` to Git!

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key (for LLM calls) | ✅ |
| `SUPABASE_URL` | Supabase project URL | When using `data.provider: "supabase"` |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | When using `data.provider: "supabase"` |
| `SUPABASE_SERVICE_KEY` | Supabase service key (for server-side access) | When using `data.provider: "supabase"` |
| `APP_DEBUG` | Enable debug mode (`true`/`false`) | ❌ (default: `false`) |

---

## Configuration (`config.yaml`)

Manage the core application settings in the `config.yaml` file.

### LLM Provider Configuration

```yaml
llm:
  provider: "openrouter" 
  model: "openai/gpt-5.2"
  temperature: 0.2
  seed: 42
  max_tokens: 128000
  max_context_tokens: 400000
  reasoning_enabled: true
  reasoning_max_tokens: 64000
  web_search_enabled: true
  web_search_max_results: 3
```

| Option | Description | Default |
|--------|-------------|---------|
| `provider` | LLM provider selection (`openrouter`, `anthropic`, `openai`) | `openrouter` |
| `model` | Model name to use | `openai/gpt-5.2` |
| `temperature` | Response creativity control (0.0 = deterministic, 1.0 = creative) | `0.2` |
| `seed` | Random seed for reproducible outputs | `42` |
| `max_tokens` | Maximum output tokens | `128000` |
| `max_context_tokens` | Context window size (input + output) | `400000` |
| `reasoning_enabled` | Enable reasoning mode for thinking models | `true` |
| `reasoning_max_tokens` | Tokens allocated for reasoning | `64000` |
| `web_search_enabled` | Enable real-time web search (OpenRouter only) | `true` |
| `web_search_max_results` | Number of web search results (1-10) | `3` |

### Data Provider Configuration

```yaml
data:
  provider: "supabase"  # supabase | local
  local_storage_path: "./data/prices"  # CSV storage path (when provider='local')
  is_paper: true
  fallback_providers: ["yfinance", "mock"]
  cache_ttl_seconds: 300
```

| Option | Description | Default |
|--------|-------------|---------|
| `provider` | Data storage selection (`supabase` or `local`) | `supabase` |
| `local_storage_path` | Local CSV storage path (used when provider='local') | `./data/prices` |
| `is_paper` | Use paper trading mode | `true` |
| `fallback_providers` | Fallback provider list when primary fails | `["yfinance", "mock"]` |
| `cache_ttl_seconds` | Data cache TTL in seconds | `300` |

**Data Provider Options:**
- `supabase`: Store price data in Supabase DB (fetched from YFinance)
- `local`: Store price data as local CSV files (fetched from YFinance)

### Execution Provider Configuration

```yaml
execution:
  provider: "docker"  # docker | local
  fallback_to_local: true
  docker_image: "backtest-runner:latest"
  docker_socket_url: null
  timeout: 300
  memory_limit: "2g"
  allowed_modules:
    - pandas
    - numpy
```

| Option | Description | Default |
|--------|-------------|---------|
| `provider` | Code execution environment selection | `docker` |
| `fallback_to_local` | Fallback to local execution if Docker fails | `true` |
| `docker_image` | Docker image name to use | `backtest-runner:latest` |
| `docker_socket_url` | Docker socket URL (null = auto-detect) | `null` |
| `timeout` | Execution timeout in seconds | `300` |
| `memory_limit` | Docker container memory limit | `2g` |
| `allowed_modules` | Allowed Python modules in backtest code | `[pandas, numpy]` |

**Execution Environment Options:**
- `docker`: Execute code securely in isolated Docker containers (recommended)
- `local`: Execute directly in local Python environment

---

## License

MIT License
