# 🚀 Oracle Cloud 배포 가이드

이 문서는 GitHub Actions를 사용하여 Oracle Cloud에 자동 배포하는 방법을 설명합니다.

## 📋 사전 준비

### 1. Oracle Cloud VM 설정

```bash
# SSH로 Oracle Cloud VM에 접속
ssh opc@<YOUR_ORACLE_VM_IP>

# Docker 설치
sudo dnf install -y dnf-utils
sudo dnf config-manager --add-repo https://download.docker.com/linux/oracle/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Docker 서비스 시작 및 활성화
sudo systemctl start docker
sudo systemctl enable docker

# 현재 사용자를 docker 그룹에 추가 (재로그인 필요)
sudo usermod -aG docker $USER

# 재로그인 후 확인
docker ps

# 방화벽 포트 열기 (Oracle Cloud Security List에서도 열어야 함)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### 2. Oracle Cloud Security List 설정

Oracle Cloud Console에서:
1. **Networking** → **Virtual Cloud Networks** → VCN 선택
2. **Security Lists** → Default Security List 선택
3. **Add Ingress Rules**:
   - Source CIDR: `0.0.0.0/0`
   - Destination Port Range: `8000`
   - Protocol: TCP

---

## 🔐 GitHub Secrets 설정

GitHub 레포지토리 → **Settings** → **Secrets and variables** → **Actions**에서 다음 시크릿을 추가:

### 필수 시크릿

| Secret Name | 설명 | 예시 |
|-------------|------|------|
| `ORACLE_HOST` | Oracle VM 공인 IP | `129.154.xxx.xxx` |
| `ORACLE_USER` | SSH 사용자명 | `opc` (Oracle Linux) 또는 `ubuntu` |
| `ORACLE_SSH_KEY` | SSH 개인키 (전체 내용) | `-----BEGIN RSA PRIVATE KEY-----...` |
| `GH_PAT` | GitHub Personal Access Token (read:packages 권한) | `ghp_xxxx...` |

### 앱 환경변수 시크릿

| Secret Name | 설명 |
|-------------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 키 |
| `SUPABASE_URL` | Supabase 프로젝트 URL (선택) |
| `SUPABASE_ANON_KEY` | Supabase Anon Key (선택) |
| `SUPABASE_SERVICE_KEY` | Supabase Service Key (선택) |
| `APP_DEBUG` | 디버그 모드 (`true` 또는 `false`) |

### GitHub PAT 생성 방법

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. **Generate new token (classic)**
3. 권한 선택:
   - `read:packages` ✅
   - `write:packages` ✅ (선택)
4. 생성된 토큰을 `GH_PAT` 시크릿으로 저장

---

## 📁 프로젝트 구조

```
backtest-simulation/
├── Dockerfile                    # 메인 API 이미지 (FastAPI)
├── docker-compose.yml           # 로컬 개발용
├── .dockerignore
├── docker/
│   └── backtest-runner/
│       └── Dockerfile           # 샌드박스 이미지 (백테스트 실행용)
├── .github/
│   └── workflows/
│       ├── ci.yml              # PR/Push 시 테스트
│       └── deploy.yml          # main 브랜치 배포
└── ...
```

---

## 🔄 배포 워크플로우

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  git push    │ ──▶ │ GitHub Actions  │ ──▶ │  Oracle Cloud    │
│  to main     │     │  CI/CD Pipeline │     │  VM Deployment   │
└──────────────┘     └─────────────────┘     └──────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ 1. Build API    │
                     │ 2. Build Runner │
                     │ 3. Push to GHCR │
                     │ 4. SSH Deploy   │
                     └─────────────────┘
```

---

## 🖥️ 로컬 테스트

```bash
# Docker 이미지 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

---

## 🛠️ 수동 배포

GitHub Actions 대신 수동으로 배포하려면:

```bash
# 1. 로컬에서 이미지 빌드
docker build -t backtest-api:latest .
docker build -f docker/backtest-runner/Dockerfile -t backtest-runner:latest .

# 2. 이미지 저장
docker save backtest-api:latest | gzip > backtest-api.tar.gz
docker save backtest-runner:latest | gzip > backtest-runner.tar.gz

# 3. Oracle VM으로 전송
scp backtest-api.tar.gz backtest-runner.tar.gz opc@<ORACLE_IP>:~

# 4. SSH 접속 후 이미지 로드 및 실행
ssh opc@<ORACLE_IP>
docker load < backtest-api.tar.gz
docker load < backtest-runner.tar.gz

# 5. 컨테이너 실행
docker run -d \
  --name backtest-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY="your_key" \
  -v ~/backtest-simulation/data:/app/data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/backtest_workspaces:/tmp/backtest_workspaces \
  --group-add $(getent group docker | cut -d: -f3) \
  backtest-api:latest
```

---

## 🔍 트러블슈팅

### Docker 권한 오류
```bash
# Docker 그룹 ID 확인
getent group docker

# 컨테이너에 그룹 추가
docker run ... --group-add <DOCKER_GID> ...
```

### 이미지 pull 실패
```bash
# GHCR 로그인 확인
echo $GH_PAT | docker login ghcr.io -u USERNAME --password-stdin

# 이미지 존재 확인
docker pull ghcr.io/OWNER/REPO:latest
```

### 헬스체크 실패
```bash
# 컨테이너 로그 확인
docker logs backtest-api

# 컨테이너 상태 확인
docker ps -a
docker inspect backtest-api
```

---

## 📊 모니터링

```bash
# 실행 중인 컨테이너 확인
docker ps

# 리소스 사용량 모니터링
docker stats backtest-api

# 로그 실시간 확인
docker logs -f backtest-api

# 헬스 체크
curl http://localhost:8000/health
```
