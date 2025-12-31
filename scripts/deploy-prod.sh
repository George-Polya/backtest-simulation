#!/bin/bash
# ===========================================
# Production Deployment Script
# ===========================================
# This script sets up the production environment
# with Nginx reverse proxy and Let's Encrypt TLS
#
# Usage: ./scripts/deploy-prod.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ===========================================
# Pre-flight checks
# ===========================================
log_info "Running pre-flight checks..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# ===========================================
# Environment setup
# ===========================================
log_info "Setting up environment..."

# Check for .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    log_warn ".env file not found. Creating from .env.example..."
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_warn "Please edit .env file with your actual values!"
    else
        log_error ".env.example not found. Please create .env manually."
        exit 1
    fi
fi

# Source .env file
source "$PROJECT_DIR/.env"

# Get Docker group ID
DOCKER_GID=$(getent group docker | cut -d: -f3 2>/dev/null || echo "999")
export DOCKER_GID

log_info "Docker GID: $DOCKER_GID"

# ===========================================
# Create required directories
# ===========================================
log_info "Creating required directories..."

mkdir -p "$PROJECT_DIR/data/prices"
mkdir -p /tmp/backtest_workspaces
chmod 777 /tmp/backtest_workspaces

# ===========================================
# DNS Check (optional but recommended)
# ===========================================
log_info "Checking DNS resolution for yoyak.org..."

DOMAIN="yoyak.org"
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

DNS_IP=$(dig +short "$DOMAIN" 2>/dev/null | head -1 || echo "")

if [ -n "$DNS_IP" ]; then
    if [ "$DNS_IP" = "$SERVER_IP" ]; then
        log_info "✓ DNS is correctly pointing to this server ($SERVER_IP)"
    else
        log_warn "DNS points to $DNS_IP but server IP is $SERVER_IP"
        log_warn "Make sure DNS is configured correctly before proceeding."
    fi
else
    log_warn "Could not resolve DNS for $DOMAIN"
    log_warn "Please ensure DNS A record points to: $SERVER_IP"
fi

# ===========================================
# Firewall check
# ===========================================
log_info "Checking firewall rules..."

if command -v ufw &> /dev/null; then
    log_info "Checking UFW status..."
    sudo ufw status | grep -E "80|443" || log_warn "Ports 80/443 may not be open in UFW"
elif command -v firewall-cmd &> /dev/null; then
    log_info "Checking firewalld status..."
    sudo firewall-cmd --list-ports | grep -E "80|443" || log_warn "Ports 80/443 may not be open in firewalld"
fi

# ===========================================
# Deploy with Docker Compose
# ===========================================
log_info "Pulling latest images..."
cd "$PROJECT_DIR"

# Use docker compose (v2) or docker-compose (v1)
COMPOSE_CMD="docker compose"
if ! docker compose version &> /dev/null; then
    COMPOSE_CMD="docker-compose"
fi

$COMPOSE_CMD -f docker-compose.prod.yml pull

log_info "Stopping existing containers..."
$COMPOSE_CMD -f docker-compose.prod.yml down --remove-orphans 2>/dev/null || true

log_info "Starting services..."
$COMPOSE_CMD -f docker-compose.prod.yml up -d

# ===========================================
# Health check
# ===========================================
log_info "Waiting for services to start..."
sleep 15

log_info "Checking container status..."
$COMPOSE_CMD -f docker-compose.prod.yml ps

log_info "Checking internal health..."
if curl -sf http://localhost:8000/health &>/dev/null; then
    log_info "✓ Backend API is healthy"
else
    log_warn "Backend health check pending..."
fi

# ===========================================
# SSL Certificate status
# ===========================================
log_info "Checking SSL certificate status..."
sleep 5

# Check if Let's Encrypt has issued a certificate
if docker exec nginx-proxy test -f /etc/nginx/certs/yoyak.org.crt 2>/dev/null; then
    log_info "✓ SSL certificate found for yoyak.org"
else
    log_warn "SSL certificate not yet issued. This may take a few minutes."
    log_warn "View acme-companion logs: docker logs acme-companion"
fi

# ===========================================
# Summary
# ===========================================
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Your site should be available at:"
echo "  - https://yoyak.org"
echo "  - https://www.yoyak.org"
echo ""
echo "Useful commands:"
echo "  - View logs: docker compose -f docker-compose.prod.yml logs -f"
echo "  - View nginx logs: docker logs nginx-proxy"
echo "  - View SSL logs: docker logs acme-companion"
echo "  - Restart: docker compose -f docker-compose.prod.yml restart"
echo ""
