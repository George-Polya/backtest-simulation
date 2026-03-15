# 로컬 실행 가이드

## 사전 준비

- Docker 및 Docker Compose 설치
- LLM API 키 (OpenRouter)

## 실행 방법

### 1. 환경변수 설정

```bash
# backend/.env 파일 생성
cp backend/.env.example backend/.env

# .env 파일에서 API 키 설정
# OPENROUTER_API_KEY=your_key_here
```

### 2. Docker Compose로 실행

```bash
# 빌드 및 실행
docker compose up --build

# 백그라운드 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f

# 중지
docker compose down
```

### 3. 접속

- API: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health

## 트러블슈팅

### Docker 소켓 권한 오류

```bash
# Docker 그룹 ID 확인 후 .env에 설정
echo "DOCKER_GID=$(getent group docker | cut -d: -f3)" >> .env
```

### 헬스체크 실패

```bash
# 컨테이너 로그 확인
docker compose logs backtest-api

# 컨테이너 상태 확인
docker compose ps
```
