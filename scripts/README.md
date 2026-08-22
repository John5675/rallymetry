# Scripts

Repository-level utilities remain thin wrappers over reusable runtime code in the
installable `services/vision` package.

- `run-local-worker.sh` loads the ignored worker environment and starts the
  single-concurrency MongoDB worker.
- `install-macos-worker.sh` installs the worker as a per-user macOS LaunchAgent so
  queued matches resume whenever this Mac is logged in.
- `com.rallymetry.worker.plist.example` is the credential-free launchd template.
