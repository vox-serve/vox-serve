#!/usr/bin/env bash
set -e

# 等价启动：不改现有脚本/源码。前台输出到终端，仅错误写入精准时间日志。

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

# 允许自定义日志时区（默认写入西雅图当地时间）。
LOG_TZ="${LOG_TZ:-America/Los_Angeles}"
if [[ -n "$LOG_TZ" ]]; then
  export TZ="$LOG_TZ"
fi

TS="$(date +%Y-%m-%dT%H-%M-%S.%3N)"
LOG_FILE="$LOG_DIR/vox-serve-$TS.err.log"
LOG_FILE_STDOUT="$LOG_DIR/vox-serve-$TS.out.log"

# 固定默认，与你当前使用保持等价；若要改端口或模型，请手动改本文件或命令行
MODEL=glm
PORT=8080
HOST=0.0.0.0
export CUDA_VISIBLE_DEVICES=0

# CUDA toolkit may not be on PATH or CPATH (no root fixes), so prefer a user-level override.
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  cuda_bin="$CUDA_HOME/bin"
  cuda_inc="$CUDA_HOME/targets/x86_64-linux/include"
  cuda_lib="$CUDA_HOME/targets/x86_64-linux/lib"
  cuda_lib_stubs="$cuda_lib/stubs"

  if [[ -d "$cuda_bin" && ":$PATH:" != *":$cuda_bin:"* ]]; then
    export PATH="$cuda_bin:$PATH"
  fi

  if [[ -d "$cuda_inc" && ":${CPLUS_INCLUDE_PATH:-}:" != *":$cuda_inc:"* ]]; then
    export CPLUS_INCLUDE_PATH="$cuda_inc${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  fi

  if [[ -d "$cuda_inc" && ":${C_INCLUDE_PATH:-}:" != *":$cuda_inc:"* ]]; then
    export C_INCLUDE_PATH="$cuda_inc${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  fi

  if [[ -d "$cuda_lib" && ":${LIBRARY_PATH:-}:" != *":$cuda_lib:"* ]]; then
    export LIBRARY_PATH="$cuda_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi

  if [[ -d "$cuda_lib_stubs" && ":${LIBRARY_PATH:-}:" != *":$cuda_lib_stubs:"* ]]; then
    export LIBRARY_PATH="$cuda_lib_stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
  fi

  if [[ -d "$cuda_lib" && ":${LD_LIBRARY_PATH:-}:" != *":$cuda_lib:"* ]]; then
    export LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

# -- 日志辅助：为 stdout / stderr 增加带时区时间戳的 tee ----------------------
timestamp_stdout() {
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '%s [stdout] %s\n' "$(date +%Y-%m-%dT%H:%M:%S.%3N%z)" "$line" \
      | tee -a "$LOG_FILE_STDOUT"
  done
}

timestamp_stderr() {
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '%s [stderr] %s\n' "$(date +%Y-%m-%dT%H:%M:%S.%3N%z)" "$line" \
      | tee -a "$LOG_FILE" >&2
  done
}

# 选择运行器：优先 uv，其次 .venv，最后系统 python
if command -v uv >/dev/null 2>&1; then
  CMD=(uv run python -m vox_serve.launch --model "$MODEL" --host "$HOST" --port "$PORT")
elif [[ -x "$DIR/.venv/bin/python" ]]; then
  CMD=("$DIR/.venv/bin/python" -m vox_serve.launch --model "$MODEL" --host "$HOST" --port "$PORT")
else
  CMD=(python -m vox_serve.launch --model "$MODEL" --host "$HOST" --port "$PORT")
fi

ln -sfn "$LOG_FILE" "$LOG_DIR/latest.err.log"
ln -sfn "$LOG_FILE_STDOUT" "$LOG_DIR/latest.out.log"
echo "Logging stderr to: $LOG_FILE"
echo "Logging stdout to: $LOG_FILE_STDOUT"

# 前台执行：stdout / stderr 均写入日志并保留终端输出。
exec \
  > >(timestamp_stdout) \
  2> >(timestamp_stderr)

exec "${CMD[@]}"
