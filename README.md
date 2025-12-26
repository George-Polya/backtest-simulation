# Natural Language Backtesting Service

> **AI가 투자 전략을 코드로 자동 생성하여 백테스팅하는 서비스**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 개요

Natural Language Backtesting Service는 사용자가 **자연어로 설명한 투자 전략을 AI가 Python 코드로 변환**하고, 이를 실행하여 백테스팅 결과를 제공하는 서비스입니다.

### 핵심 기능

- 🤖 **AI 기반 코드 생성**: OpenRouter, Claude, GPT 등 LLM을 활용하여 자연어 전략을 Python 코드로 변환
- 📊 **실시간 데이터**: Yahoo Finance를 통한 글로벌 주식 데이터 제공
- 🔒 **안전한 실행**: Docker 샌드박스에서 안전하게 코드 실행
- 📈 **시각화**: 수익률 그래프, 성과 지표, 생성된 코드 확인 가능
- 🎯 **SOLID 원칙 준수**: 확장 가능하고 유지보수하기 쉬운 아키텍처

## 빠른 시작

### 1. 환경 설정

```bash
# Python 3.11 이상 필요
# 저장소 클론
git clone https://github.com/yourusername/open-trading-api.git
cd open-trading-api

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정 파일 구성

```bash
# 환경변수 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
```

`.env` 파일에 LLM API 키를 입력하세요:

```env
# LLM API Keys (choose one)
OPENROUTER_API_KEY=your_openrouter_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional
APP_DEBUG=false
```

`config.yaml` 파일을 확인하고 필요 시 수정하세요:

```yaml
# LLM Provider Configuration
llm:
  provider: "openrouter"  # openrouter | anthropic | openai
  model: "openai/gpt-4o"
  temperature: 0.2
  max_tokens: 8000

# Data Provider Configuration
data:
  provider: "yfinance"  # 글로벌 주식 데이터
  fallback_providers: ["mock"]

# Code Execution Configuration
execution:
  provider: "docker"  # docker | local
  fallback_to_local: true
  timeout: 300
```

### 3. 서버 실행

```bash
# FastAPI 서버 시작
uvicorn app.main:app --reload --port 8000

# 또는 Python으로 직접 실행
python -m app.main
```

서버가 시작되면 다음 주소로 접속:
- API 문서: http://localhost:8000/docs
- 대시보드: http://localhost:8000/dashboard/
- Health Check: http://localhost:8000/health

## 사용 예시

### API를 통한 백테스팅

```bash
curl -X POST "http://localhost:8000/api/v1/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_description": "SPY를 60%, QQQ를 40% 비율로 매수하고 보유하는 전략",
    "tickers": ["SPY", "QQQ"],
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_cash": 10000
  }'
```

### Python 클라이언트

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/backtest",
    json={
        "strategy_description": "5일 이동평균이 20일 이동평균을 상향 돌파하면 매수, 하향 돌파하면 매도",
        "tickers": ["AAPL"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_cash": 10000
    }
)

result = response.json()
print(f"총 수익률: {result['metrics']['total_return']}%")
print(f"샤프 비율: {result['metrics']['sharpe_ratio']}")
```

## 프로젝트 구조

```
open-trading-api/
├── app/
│   ├── api/                    # FastAPI 엔드포인트
│   │   └── v1/
│   │       └── endpoints/
│   │           └── backtest.py
│   ├── core/                   # 핵심 설정 및 DI 컨테이너
│   │   ├── config.py
│   │   └── container.py
│   ├── providers/              # 외부 서비스 어댑터
│   │   ├── llm/               # LLM 프로바이더 (OpenRouter, Claude, GPT)
│   │   └── data/              # 데이터 프로바이더 (YFinance, Mock)
│   ├── services/              # 비즈니스 로직
│   │   ├── code_generator.py  # AI 코드 생성
│   │   ├── code_validator.py  # 코드 검증
│   │   └── execution/         # 코드 실행 (Docker/Local)
│   ├── dashboard/             # Dash 대시보드
│   └── main.py                # FastAPI 애플리케이션
├── tests/                     # 테스트
├── docker/                    # Docker 설정
├── config.yaml                # 앱 설정
├── .env                       # 환경변수 (비공개)
└── requirements.txt           # Python 의존성
```

## 아키텍처

### SOLID 원칙 기반 설계

- **단일 책임 원칙 (SRP)**: 각 모듈은 하나의 명확한 책임만 가짐
- **개방-폐쇄 원칙 (OCP)**: 새로운 LLM/데이터 제공자 추가 시 기존 코드 수정 불필요
- **리스코프 치환 원칙 (LSP)**: 모든 어댑터는 동일한 인터페이스로 교체 가능
- **인터페이스 분리 원칙 (ISP)**: 최소한의 인터페이스만 정의
- **의존성 역전 원칙 (DIP)**: 추상화에 의존하며, DI 컨테이너를 통해 주입

### 주요 컴포넌트

```
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Application                    │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Code         │  │ LLM          │  │ Data         │  │
│  │ Generator    │─>│ Provider     │  │ Provider     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                 │                  │          │
│         ▼                 ▼                  ▼          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Code         │  │ OpenRouter/  │  │ YFinance/    │  │
│  │ Validator    │  │ Claude/GPT   │  │ Mock         │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Execution Manager (Docker/Local)         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 지원하는 기능

### LLM 프로바이더

- ✅ OpenRouter (권장: 다양한 모델 지원)
- ✅ Anthropic Claude
- ✅ OpenAI GPT
- ✅ Web Search (OpenRouter)

### 데이터 소스

- ✅ Yahoo Finance (글로벌 주식)
- ✅ Mock (테스트용)
- ✅ 로컬 캐싱 (빠른 백테스팅)

### 실행 환경

- ✅ Docker (샌드박스, 권장)
- ✅ Local (빠른 개발)

## 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 확인
pytest --cov=app --cov-report=html

# 특정 테스트만 실행
pytest tests/api/v1/test_backtest_e2e.py -v
```

## Docker를 사용한 실행

```bash
# Docker 이미지 빌드
docker build -f docker/backtest-runner/Dockerfile -t backtest-runner:latest .

# Docker Compose로 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f app
```

## 개발 가이드

### 새로운 LLM 프로바이더 추가

1. `app/providers/llm/` 에 새 어댑터 클래스 생성
2. `LLMProvider` 인터페이스 구현
3. `app/providers/llm/factory.py` 에 등록
4. `config.yaml` 에 설정 추가

### 새로운 데이터 소스 추가

1. `app/providers/data/` 에 새 어댑터 클래스 생성
2. `DataProvider` 인터페이스 구현
3. `app/providers/data/factory.py` 에 등록
4. `config.yaml` 에 설정 추가

## 문제 해결

### Docker 연결 오류

```bash
# Docker 소켓 경로 확인
ls -la /var/run/docker.sock  # Linux/WSL
ls -la ~/.docker/run/docker.sock  # macOS

# config.yaml에 명시적으로 설정
execution:
  docker_socket_url: "unix:///var/run/docker.sock"
```

### LLM API 오류

```bash
# API 키 확인
cat .env | grep API_KEY

# 프로바이더 변경
# config.yaml에서 provider를 변경하세요
llm:
  provider: "anthropic"  # openrouter에서 변경
```

## 기여

기여를 환영합니다! Pull Request를 보내주세요.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 연락처

프로젝트 관련 문의: [이슈 페이지](https://github.com/yourusername/open-trading-api/issues)

---

**⚠️ 면책 조언**: 이 소프트웨어는 교육 및 연구 목적으로 제공됩니다. 실제 투자 전에 충분한 검토가 필요하며, 투자 손실에 대한 책임은 사용자에게 있습니다.
