#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
worker_env="${repo_root}/.env.worker"
template="${script_dir}/com.rallymetry.worker.plist.example"
support_dir="${HOME}/Library/Application Support/Rallymetry"
installed_env="${support_dir}/worker.env"
installed_plan="${support_dir}/pipeline-plan.json"
installed_runner="${support_dir}/run-installed-worker.sh"
launch_agents="${HOME}/Library/LaunchAgents"
destination="${launch_agents}/com.rallymetry.worker.plist"
log_dir="${HOME}/Library/Logs"

if [[ ! -f "${worker_env}" ]]; then
  print -u2 "Missing ${worker_env}. Copy .env.worker.example and add private credentials first."
  exit 2
fi

chmod 600 "${worker_env}"
mkdir -p "${launch_agents}" "${log_dir}" "${support_dir}/tmp"

# launchd cannot reliably read repositories under macOS-protected Documents.
# Install a self-contained user tool and copy only credential-free runtime files
# plus the already-ignored worker environment into Application Support.
uv tool install --force "${repo_root}/services/vision"
install -m 700 "${script_dir}/run-installed-worker.sh" "${installed_runner}"
install -m 600 "${worker_env}" "${installed_env}"
install -m 600 \
  "${repo_root}/docs/examples/render-workflow-pipeline-plan.json" \
  "${installed_plan}"
sed -i '' \
  -e "s|^PIPELINE_CONFIG=.*|PIPELINE_CONFIG='${installed_plan}'|" \
  -e "s|^WORKFLOW_TEMP_DIR=.*|WORKFLOW_TEMP_DIR='${support_dir}/tmp'|" \
  "${installed_env}"

sed \
  -e "s|__SUPPORT_DIR__|${support_dir}|g" \
  -e "s|__LOG_DIR__|${log_dir}|g" \
  "${template}" > "${destination}"
plutil -lint "${destination}"

launchctl bootout "gui/${UID}" "${destination}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${UID}" "${destination}"
launchctl kickstart -k "gui/${UID}/com.rallymetry.worker"

print "Rallymetry worker installed and started."
print "Logs: ${log_dir}/rallymetry-worker.log"
