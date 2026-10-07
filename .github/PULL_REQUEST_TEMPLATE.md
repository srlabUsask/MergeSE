<!--
Thanks for the contribution. The CI pipeline will run:
  * ruff + pyflakes
  * JS syntax check on frontend/app.js
  * presets.json schema sanity (no fs paths)
  * pytest on Python 3.10 / 3.11 / 3.12 with coverage >= 55%
  * wheel install smoke (if you touched packaging / server / frontend)
  * docker build + boot (if you touched the Dockerfile / deploy)
  * pip-audit + secret tripwire

A green CI is required before merge; see .github/workflows/*.yml.
-->

### Summary

<!-- One paragraph: what changed and why. Link the audit ID / issue number if any. -->

### Risk + rollback

<!-- Is this user-facing on mergese.usask.ca? What does `git revert` look like if something regresses in prod? -->

### Testing

- [ ] `pytest` passes locally
- [ ] If touching the server: smoke-tested with `python server/app.py` and a `curl` against the changed endpoint
- [ ] If touching the UI: hard-refreshed the browser and clicked through
- [ ] If touching presets/data: verified the preset validator in `.github/workflows/ci.yml` still passes

### Deploy note (fill if the merge needs an operator action beyond `git pull && systemctl restart`)

<!-- e.g. new env var, pip install of a new dep, rsync of new checkpoints -->
