#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
worker_env="${repo_root}/.env.worker"
template="${script_dir}/com.rallymetry.worker.plist.example"
launch_agents="${HOME}/Library/LaunchAgents"
destination="${launch_agents}/com.rallymetry.worker.plist"

if [[ ! -f "${worker_env}" ]]; then
  print -u2 "Missing ${worker_env}. Copy .env.worker.example and add private credentials first."
  exit 2
fi

chmod 600 "${worker_env}"
mkdir -p "${launch_agents}" "${HOME}/Library/Logs"
sed \
  -e "s|__REPO_ROOT__|${repo_root}|g" \
  -e "s|__HOME__|${HOME}|g" \
  "${template}" > "${destination}"
plutil -lint "${destination}"

launchctl bootout "gui/${UID}" "${destination}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/${UID}" "${destination}"
launchctl kickstart -k "gui/${UID}/com.rallymetry.worker"

print "Rallymetry worker installed and started."
print "Logs: ${HOME}/Library/Logs/rallymetry-worker.log"
