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
git clone https://github.com/yourusername/backtest-simulation.git
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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_KEY` | Supabase secret key |

## License

MIT License
