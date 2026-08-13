# Pickleball Vision Milestones

Milestones are intentionally sequential. Exactly one milestone is current. Work
outside the current milestone requires this plan to be deliberately updated first.

## 0. Foundation — current

Establish repository instructions, architecture documentation, a typed Python
package, an installable CLI shell, environment-based configuration, structured
logging, error conventions, and automated quality checks.

Exit criteria:

- The documented repository structure exists.
- The vision package installs in a clean environment.
- `pickleball-vision doctor` reports a valid Foundation setup.
- Tests, Ruff checks, Ruff formatting, and static type checks pass.
- No video ingestion or computer-vision behavior is implemented.

## Future milestones — not current

1. Video ingestion
2. Court calibration
3. Person detection
4. Primary-player isolation
5. Persistent player tracking
6. Player court-position analytics
7. Ball dataset tooling
8. Ball detector
9. Ball trajectory reconstruction
10. Rally annotation
11. Rally segmentation
12. Bounce detection
13. Contact detection
14. Hitter identification
15. Shot classification
16. Analytics engine
17. Backend productization
18. Async processing
19. Web dashboard
20. Human correction
21. AI coach

Moving a milestone to current should add measurable entry and exit criteria while
preserving completed milestone history. Do not bundle later milestones into the
current one for convenience.
