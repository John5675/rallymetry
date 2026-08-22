#!/bin/zsh
set -euo pipefail

support_dir="${0:A:h}"
worker_env="${support_dir}/worker.env"

if [[ ! -f "${worker_env}" ]]; then
  print -u2 "Missing installed worker environment: ${worker_env}"
  exit 2
fi

set -a
source "${worker_env}"
set +a

# launchd supplies only a minimal environment. The orchestrator intentionally
# resolves the trusted CLI by name before starting each isolated pipeline stage,
# so include the user UV tool and common package-manager locations explicitly.
export PATH="${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

exec "${HOME}/.local/bin/pickleball-vision" worker
