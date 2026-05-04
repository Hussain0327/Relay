#!/usr/bin/env bash
# Relay Milestone 1: bring up vLLM serving Llama 3 8B Instruct on a RunPod L4.
# HF token: https://huggingface.co/settings/tokens   |   Llama 3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# Idempotent: re-running detects existing driver, Docker runtime, image, container, and weights, and skips.

set -euo pipefail

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-relay-vllm}"
HOST_PORT="${HOST_PORT:-8001}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/root/.cache/huggingface}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-1200}"

log()  { printf '[runpod_setup] %s\n' "$*"; }
fail() { printf '[runpod_setup] ERROR: %s\n' "$*" >&2; exit 1; }

# -------- 1. Required environment --------
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN unset. Create a token at https://huggingface.co/settings/tokens and accept the Llama 3 license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct, then export HF_TOKEN=hf_xxx and re-run."

# -------- 2. NVIDIA driver --------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver present: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  fail "nvidia-smi unavailable. RunPod GPU images ship with drivers; verify the pod is GPU-backed."
fi

# -------- 3. Docker --------
if command -v docker >/dev/null 2>&1; then
  log "Docker present: $(docker --version)"
else
  log "Installing Docker via get.docker.com..."
  curl -fsSL https://get.docker.com | sh
fi

# -------- 4. NVIDIA Container Toolkit --------
if docker info 2>/dev/null | grep -Eq 'Runtimes:.*\bnvidia\b'; then
  log "NVIDIA container runtime already configured."
else
  log "Installing NVIDIA Container Toolkit (apt-based, expects Ubuntu/Debian)..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# -------- 5. HF cache volume (host-mounted for warm restarts) --------
mkdir -p "$HF_CACHE_DIR"
log "HF cache directory: $HF_CACHE_DIR"

# -------- 6. vLLM image --------
if docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  log "vLLM image already pulled: $VLLM_IMAGE"
else
  log "Pulling $VLLM_IMAGE (this can take a few minutes)..."
  docker pull "$VLLM_IMAGE"
fi

# -------- 7. Container lifecycle --------
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Container $CONTAINER_NAME already running on port ${HOST_PORT}. Skipping launch."
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Container $CONTAINER_NAME exists but is stopped. Restarting."
  docker start "$CONTAINER_NAME" >/dev/null
else
  log "Launching $CONTAINER_NAME with model $MODEL..."
  docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    -p "${HOST_PORT}:8001" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    --restart unless-stopped \
    "$VLLM_IMAGE" \
    --model "$MODEL" \
    --port 8001 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    >/dev/null
fi

# -------- 8. Wait for /health --------
log "Waiting for vLLM /health (cold start with weight download can take 5-15 min; warm start ~30s)..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
until curl -fsS "http://localhost:${HOST_PORT}/health" >/dev/null 2>&1; do
  if (( $(date +%s) > deadline )); then
    fail "vLLM did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s. Inspect: docker logs ${CONTAINER_NAME}"
  fi
  sleep 5
done
log "vLLM healthy."

# -------- 9. Smoke test --------
SMOKE_OUT="/tmp/relay-m1-smoke.json"
log "Smoke test: POST /v1/chat/completions -> $SMOKE_OUT"
curl -fsS "http://localhost:${HOST_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Say hello in five words."}],
  "max_tokens": 32,
  "temperature": 0
}
JSON
)" | tee "$SMOKE_OUT"
echo
log "Milestone 1 complete. Smoke output: $SMOKE_OUT"
