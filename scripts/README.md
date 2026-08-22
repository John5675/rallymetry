# Scripts

Repository-level utilities remain thin wrappers over reusable runtime code in the
installable `services/vision` package.

- `run-local-worker.sh` loads the ignored worker environment and starts the
  single-concurrency MongoDB worker.
- `install-macos-worker.sh` installs the worker as a per-user macOS LaunchAgent so
  queued matches resume whenever this Mac is logged in. It installs an isolated UV
  tool and runtime files under `~/Library/Application Support/Rallymetry` so launchd
  does not need access to the protected `Documents` directory.
- `run-installed-worker.sh` is the small Application Support launcher used by
  launchd after installation.
- `com.rallymetry.worker.plist.example` is the credential-free launchd template.
