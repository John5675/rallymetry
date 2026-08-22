#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
worker_env="${RALLYMETRY_WORKER_ENV:-${repo_root}/.env.worker}"

if [[ ! -f "${worker_env}" ]]; then
  print -u2 "Missing worker environment: ${worker_env}"
  print -u2 "Copy .env.worker.example to .env.worker and add private credentials."
  exit 2
fi

# A Vercel CLI pull already stores Blob tokens in this ignored file. Loading it
# first avoids copying those credentials into another file; .env.worker can still
# override any value explicitly.
if [[ -f "${repo_root}/.env.local" ]]; then
  set -a
  source "${repo_root}/.env.local"
  set +a
fi

set -a
source "${worker_env}"
set +a

cd "${repo_root}/services/vision"
exec uv run pickleball-vision worker "$@"
