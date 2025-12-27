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

# 데이터 제공자: supabase | yfinance | mock
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

## 환경 변수

| 변수명 | 설명 |
|--------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 키 |
| `SUPABASE_URL` | Supabase 프로젝트 URL |
| `SUPABASE_ANON_KEY` | Supabase 익명 키 |
| `SUPABASE_SERVICE_KEY` | Supabase 시크릿 키 |

## 라이선스

MIT License
