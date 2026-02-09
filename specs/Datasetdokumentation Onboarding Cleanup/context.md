# Context

Review found onboarding blockers in `Datasetdokumentation/`:
- Build script invokes `sass` but package manifest did not declare it.
- README pointed to an example file path that does not exist in this repo.
- README lacked a concrete install-first sequence for a new clone.
- A 0-byte `.print.pdf` artifact was tracked, which can mislead contributors.
- Lockfiles were ignored, weakening reproducibility for JavaScript dependencies.

This task focuses on contributor-first setup clarity and deterministic project bootstrap behavior.
