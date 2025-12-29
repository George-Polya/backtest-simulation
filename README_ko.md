# 자연어 백테스팅 서비스

자연어로 작성된 트레이딩 전략을 실행 가능한 Python 코드로 변환하는 AI 기반 백테스팅 서비스입니다.

## 주요 기능

- **자연어 전략 입력**: 일반 언어로 트레이딩 전략 설명
- **AI 코드 생성**: LLM을 활용한 백테스팅 코드 자동 생성
- **인터랙티브 대시보드**: Plotly 차트로 백테스트 결과 시각화
- **다양한 데이터 제공자**: Supabase, YFinance, 모의 데이터 지원
- **Docker 실행**: 격리된 Docker 컨테이너에서 안전한 코드 실행

## 요구 사항

- Python 3.13+
- Docker (선택사항, 안전한 코드 실행용)

## 설치

1. 저장소 클론:
```bash
git clone https://github.com/George-Polya/backtest-simulation.git
cd backtest-simulation
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 열어 API 키 입력
```

4. (선택사항) Docker 이미지 빌드:
```bash
cd docker
docker build -t backtest-runner:latest .
```

## 설정

`config.yaml` 파일을 수정하여 설정을 변경할 수 있습니다:

```yaml
# LLM 제공자: openrouter 
llm:
  provider: "openrouter"
  model: "openai/gpt-5.2"

data:
  provider: "supabase"

# 실행 환경: docker | local
execution:
  provider: "docker"
```

## 실행

대시보드 시작:
```bash
uvicorn -m app.main:app --port 8050
```

대시보드는 `http://localhost:8050`에서 확인할 수 있습니다.

## 환경 변수 (`.env.example`)

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고, 필요한 API 키와 설정값을 입력하세요.

```bash
cp .env.example .env
```

> ⚠️ **주의**: `.env` 파일은 절대 Git에 커밋하지 마세요!

| 변수 | 설명 | 필수 |
|------|------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 키 (LLM 호출용) | ✅ |
| `SUPABASE_URL` | Supabase 프로젝트 URL | `data.provider: "supabase"` 사용 시 |
| `SUPABASE_ANON_KEY` | Supabase 익명 키 | `data.provider: "supabase"` 사용 시 |
| `SUPABASE_SERVICE_KEY` | Supabase 서비스 키 (서버 사이드 접근용) | `data.provider: "supabase"` 사용 시 |
| `APP_DEBUG` | 디버그 모드 활성화 (`true`/`false`) | ❌ (기본값: `false`) |

---

## 상세 설정 (`config.yaml`)

`config.yaml` 파일에서 애플리케이션의 핵심 설정을 관리합니다.

### LLM Provider 설정

```yaml
llm:
  provider: "openrouter"  # openrouter | anthropic | openai
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

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `provider` | LLM 제공자 선택 (`openrouter`, `anthropic`, `openai`) | `openrouter` |
| `model` | 사용할 모델명 | `openai/gpt-5.2` |
| `temperature` | 응답의 창의성 조절 (0.0 = 결정적, 1.0 = 창의적) | `0.2` |
| `seed` | 재현 가능한 출력을 위한 랜덤 시드 | `42` |
| `max_tokens` | 최대 출력 토큰 수 | `128000` |
| `max_context_tokens` | 컨텍스트 윈도우 크기 (입력+출력) | `400000` |
| `reasoning_enabled` | 사고(reasoning) 모드 활성화 | `true` |
| `reasoning_max_tokens` | 사고에 할당되는 토큰 수 | `64000` |
| `web_search_enabled` | 실시간 웹 검색 활성화 (OpenRouter만 지원) | `true` |
| `web_search_max_results` | 웹 검색 결과 수 (1-10) | `3` |

### Data Provider 설정

```yaml
data:
  provider: "supabase"  # supabase | local
  local_storage_path: "./data/prices"  # CSV 저장 경로 (provider='local' 사용 시)
  is_paper: true
  fallback_providers: ["yfinance", "mock"]
  cache_ttl_seconds: 300
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `provider` | 데이터 저장소 선택 (`supabase` 또는 `local`) | `supabase` |
| `local_storage_path` | 로컬 CSV 저장 경로 (provider='local' 사용 시) | `./data/prices` |
| `is_paper` | 페이퍼 트레이딩 모드 사용 여부 | `true` |
| `fallback_providers` | 기본 제공자 실패 시 대체 제공자 목록 | `["yfinance", "mock"]` |
| `cache_ttl_seconds` | 데이터 캐시 유효 시간 (초) | `300` |

**데이터 저장소 옵션:**
- `supabase`: YFinance에서 가져온 가격 데이터를 Supabase DB에 저장
- `local`: YFinance에서 가져온 가격 데이터를 로컬 CSV 파일에 저장

### Execution Provider 설정

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

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `provider` | 코드 실행 환경 선택 | `docker` |
| `fallback_to_local` | Docker 실패 시 로컬 실행으로 폴백 | `true` |
| `docker_image` | 사용할 Docker 이미지명 | `backtest-runner:latest` |
| `docker_socket_url` | Docker 소켓 URL (null = 자동 감지) | `null` |
| `timeout` | 실행 타임아웃 (초) | `300` |
| `memory_limit` | Docker 컨테이너 메모리 제한 | `2g` |
| `allowed_modules` | 백테스트 코드에서 허용되는 Python 모듈 | `[pandas, numpy]` |

**실행 환경 옵션:**
- `docker`: 격리된 Docker 컨테이너에서 안전하게 코드 실행 (권장)
- `local`: 로컬 Python 환경에서 직접 실행

---

## 라이선스

MIT License
