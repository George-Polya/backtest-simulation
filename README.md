# AI Backtest

**Describe your investment strategy in natural language, and AI generates Python code to backtest it.**

[English](#english) | [한국어](#한국어)

---

<a id="english"></a>

## Overview

AI Backtest is an open-source service that converts natural language investment strategies into executable Python backtesting code using a LangGraph agent workflow. Simply describe your strategy (e.g., *"Buy AAPL when it drops below the 50-day moving average"*), and the system will generate, validate, and execute the backtest in a sandboxed Docker environment.

### Key Features

- **Natural Language Input** - Describe strategies in plain English; no coding required
- **LLM Agent Code Generation** - LangGraph agent via OpenRouter with built-in API reference, code pattern, validation, and ticker extraction tools
- **Sandboxed Execution** - Backtest code runs in isolated Docker containers (DooD architecture) with memory limits
- **AST-based Validation** - Generated code is statically analyzed for security (banned imports/functions) before execution
- **Interactive Results** - Equity curves, drawdown charts, monthly return heatmaps, and trade logs
- **Code Editor** - Review and edit generated code with Monaco Editor before execution
- **Market Data** - Real-time data via yfinance with LRU caching and CSV-based local cache
- **Async Execution** - Submit backtests asynchronously and poll for results

### Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Python 3.13, FastAPI, LangChain, LangGraph |
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS |
| State | Zustand, React Query |
| Forms | React Hook Form, Zod |
| Visualization | Recharts, Monaco Editor |
| Execution | Docker (sandboxed containers, DooD) |
| Data | yfinance, pandas, numpy, Backtesting.py |

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- An [OpenRouter](https://openrouter.ai/) API key

### 1. Clone the repository

```bash
git clone https://github.com/George-Polya/backtest-simulation.git
cd backtest-simulation
```

### 2. Set up environment variables

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and add your API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

### 4. Open the app

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Running Without Docker

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
cp .env.example .env       # Edit .env with your API key
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

## Configuration

### LLM Provider

Edit `backend/config.yaml` to change the LLM provider, model, and agent runtime:

```yaml
llm:
  provider: "langchain"
  model:
    name: "openai/gpt-5.4"
    max_context_tokens: 1000000
    max_output_tokens: 128000
    reasoning_effort: "low"
  temperature: 0.0
  seed: 42
  reasoning_enabled: true
  web_search_enabled: true
  web_search_max_results: 3
  agent_max_iterations: 4
  agent_timeout_seconds: 1800
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (used by LangChain provider) |
| `DOCKER_SOCKET_URL` | No | Docker socket path (auto-detected if not set) |
| `APP_DEBUG` | No | Enable debug mode (default: `false`) |

## Project Structure

```
backtest-simulation/
├── backend/
│   ├── api/v1/              # API endpoints
│   ├── core/                # Config, DI container
│   ├── providers/
│   │   ├── llm/             # LLM providers (LangChain/OpenRouter)
│   │   └── data/            # Data providers (yfinance, local, mock)
│   ├── services/
│   │   ├── code_generation/
│   │   │   └── graph/       # LangGraph agent (nodes, state, callbacks)
│   │   └── execution/       # Job manager, Docker/local backends
│   ├── models/              # Pydantic models & error codes
│   ├── prompts/             # LLM prompt templates & API references
│   ├── utils/               # Ticker extraction utilities
│   ├── config.yaml          # Application config
│   └── main.py              # FastAPI entry point
├── frontend/
│   └── src/
│       ├── app/             # Next.js pages (App Router)
│       ├── components/      # React components (backtest, results, ui)
│       ├── hooks/           # Custom hooks (generate, execute, polling)
│       ├── stores/          # Zustand state management
│       ├── lib/
│       │   ├── api/         # Axios API client
│       │   ├── validations/ # Zod form schemas
│       │   └── export/      # CSV/JSON export
│       └── types/           # TypeScript type definitions
├── docker/
│   └── backtest-runner/     # Sandbox execution image
├── tests/                   # pytest test suite
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/backtest/generate` | Generate backtest code from natural language |
| `POST` | `/api/v1/backtest/execute` | Execute generated code (sync/async) |
| `GET` | `/api/v1/backtest/status/{job_id}` | Poll job status |
| `GET` | `/api/v1/backtest/result/{job_id}` | Get execution result |
| `GET` | `/api/v1/backtest/{job_id}/result` | Get formatted backtest result |
| `GET` | `/api/v1/backtest/{job_id}/chart/equity` | Get equity curve chart data |
| `GET` | `/api/v1/backtest/{job_id}/chart/drawdown` | Get drawdown chart data |
| `GET` | `/api/v1/backtest/{job_id}/chart/monthly-returns` | Get monthly returns heatmap data |
| `GET` | `/api/v1/backtest/config/llm-providers` | List available LLM providers |
| `GET` | `/api/v1/backtest/config/data-sources` | List available data sources |

### Generate Response Example

```json
{
  "generated_code": {
    "code": "def run_backtest(params): ...",
    "strategy_summary": "Momentum strategy over SPY and QQQ.",
    "model_info": {
      "provider": "langchain",
      "model_id": "openai/gpt-5.4",
      "max_tokens": 128000,
      "supports_system_prompt": true
    },
    "tickers": ["QQQ", "SPY"]
  },
  "tickers_found": ["SPY"],
  "generation_time_seconds": 2.4
}
```

`generated_code.tickers` is the final merged ticker list used by the generated strategy. `tickers_found` is the earlier request-level extraction result that is still returned for compatibility.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

<a id="한국어"></a>

## 개요

AI Backtest는 자연어로 작성한 투자 전략을 LangGraph 에이전트 워크플로를 통해 실행 가능한 Python 백테스팅 코드로 변환하는 오픈소스 서비스입니다. 전략을 설명하기만 하면 (예: *"AAPL이 50일 이동평균선 아래로 떨어지면 매수"*), 시스템이 코드를 생성하고 검증한 뒤 격리된 Docker 환경에서 백테스트를 실행합니다.

### 주요 기능

- **자연어 입력** - 일반 언어로 전략을 설명; 코딩 불필요
- **LLM 에이전트 코드 생성** - LangGraph 에이전트 + OpenRouter 기반 생성, API 레퍼런스/코드 패턴/검증/티커 추출 도구 내장
- **샌드박스 실행** - 보안을 위해 격리된 Docker 컨테이너에서 코드 실행 (DooD 아키텍처, 메모리 제한)
- **AST 기반 검증** - 실행 전 생성된 코드를 정적 분석하여 보안 위반 (금지된 import/함수) 탐지
- **인터랙티브 결과** - 수익률 곡선, 드로다운 차트, 월별 수익률 히트맵, 거래 로그
- **코드 에디터** - Monaco Editor로 생성된 코드를 실행 전 확인 및 수정 가능
- **시장 데이터** - yfinance를 통한 실시간 데이터, LRU 캐싱 및 CSV 기반 로컬 캐시
- **비동기 실행** - 백테스트를 비동기로 제출하고 결과를 폴링

### 기술 스택

| 레이어 | 스택 |
|--------|------|
| 백엔드 | Python 3.13, FastAPI, LangChain, LangGraph |
| 프론트엔드 | Next.js 15, React 19, TypeScript, Tailwind CSS |
| 상태 관리 | Zustand, React Query |
| 폼 처리 | React Hook Form, Zod |
| 시각화 | Recharts, Monaco Editor |
| 실행 환경 | Docker (샌드박스 컨테이너, DooD) |
| 데이터 | yfinance, pandas, numpy, Backtesting.py |

## 빠른 시작

### 사전 준비

- [Docker](https://docs.docker.com/get-docker/) 및 Docker Compose
- [OpenRouter](https://openrouter.ai/) API 키

### 1. 레포지토리 클론

```bash
git clone https://github.com/George-Polya/backtest-simulation.git
cd backtest-simulation
```

### 2. 환경변수 설정

```bash
cp backend/.env.example backend/.env
```

`backend/.env` 파일을 열어 API 키를 입력합니다:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Docker Compose로 실행

```bash
docker compose up --build
```

### 4. 앱 접속

- **프론트엔드**: http://localhost:3000
- **API 문서**: http://localhost:8000/docs
- **헬스체크**: http://localhost:8000/health

## Docker 없이 실행하기

### 백엔드

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
cp .env.example .env       # .env에 API 키 입력
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 프론트엔드

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

## 설정

### LLM 프로바이더

`backend/config.yaml`에서 LLM 프로바이더, 모델, 에이전트 런타임을 변경할 수 있습니다:

```yaml
llm:
  provider: "langchain"
  model:
    name: "openai/gpt-5.4"
    max_context_tokens: 1000000
    max_output_tokens: 128000
    reasoning_effort: "low"
  temperature: 0.0
  seed: 42
  reasoning_enabled: true
  web_search_enabled: true
  web_search_max_results: 3
  agent_max_iterations: 4
  agent_timeout_seconds: 1800
```

### 환경변수

| 변수명 | 필수 | 설명 |
|--------|------|------|
| `OPENROUTER_API_KEY` | 예 | OpenRouter API 키 (LangChain 프로바이더가 사용) |
| `DOCKER_SOCKET_URL` | 아니오 | Docker 소켓 경로 (미설정 시 자동 감지) |
| `APP_DEBUG` | 아니오 | 디버그 모드 활성화 (기본값: `false`) |

## 프로젝트 구조

```
backtest-simulation/
├── backend/
│   ├── api/v1/              # API 엔드포인트
│   ├── core/                # 설정, DI 컨테이너
│   ├── providers/
│   │   ├── llm/             # LLM 프로바이더 (LangChain/OpenRouter)
│   │   └── data/            # 데이터 프로바이더 (yfinance, local, mock)
│   ├── services/
│   │   ├── code_generation/
│   │   │   └── graph/       # LangGraph 에이전트 (노드, 상태, 콜백)
│   │   └── execution/       # 작업 관리자, Docker/로컬 백엔드
│   ├── models/              # Pydantic 모델 및 에러 코드
│   ├── prompts/             # LLM 프롬프트 템플릿 및 API 레퍼런스
│   ├── utils/               # 티커 추출 유틸리티
│   ├── config.yaml          # 애플리케이션 설정
│   └── main.py              # FastAPI 진입점
├── frontend/
│   └── src/
│       ├── app/             # Next.js 페이지 (App Router)
│       ├── components/      # React 컴포넌트 (backtest, results, ui)
│       ├── hooks/           # 커스텀 훅 (생성, 실행, 폴링)
│       ├── stores/          # Zustand 상태 관리
│       ├── lib/
│       │   ├── api/         # Axios API 클라이언트
│       │   ├── validations/ # Zod 폼 스키마
│       │   └── export/      # CSV/JSON 내보내기
│       └── types/           # TypeScript 타입 정의
├── docker/
│   └── backtest-runner/     # 샌드박스 실행 이미지
├── tests/                   # pytest 테스트 스위트
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/v1/backtest/generate` | 자연어로부터 백테스트 코드 생성 |
| `POST` | `/api/v1/backtest/execute` | 생성된 코드 실행 (동기/비동기) |
| `GET` | `/api/v1/backtest/status/{job_id}` | 작업 상태 조회 |
| `GET` | `/api/v1/backtest/result/{job_id}` | 실행 결과 조회 |
| `GET` | `/api/v1/backtest/{job_id}/result` | 포맷된 백테스트 결과 조회 |
| `GET` | `/api/v1/backtest/{job_id}/chart/equity` | 수익률 곡선 차트 데이터 |
| `GET` | `/api/v1/backtest/{job_id}/chart/drawdown` | 드로다운 차트 데이터 |
| `GET` | `/api/v1/backtest/{job_id}/chart/monthly-returns` | 월별 수익률 히트맵 데이터 |
| `GET` | `/api/v1/backtest/config/llm-providers` | 사용 가능한 LLM 프로바이더 목록 |
| `GET` | `/api/v1/backtest/config/data-sources` | 사용 가능한 데이터 소스 목록 |

## 기여하기

기여를 환영합니다! Pull Request를 자유롭게 제출해주세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
