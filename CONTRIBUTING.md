# Contributing

## Setup

```bash
git clone https://github.com/srlabUsask/MergeSE.git
cd MergeSE
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install ".[dev]"
```

Check the install with `pytest -q` and `mergese --help`.

For the web tool add `pip install ".[server,datasets]"` and run
`python server/app.py`.

## Bugs

Use the bug report template. Please include the exact CLI or API call, the
error, and the output of `pip show mergese torch transformers`.

## Small PRs

If the change is a small fix, a docs improvement, or a new task
registration in `mergese_tasks.py`, just open a PR. Please:

- run `pytest -q` first (CI runs it on 3.10 and 3.11)
- add a test if you changed behaviour
- reference the issue in the PR body

## Larger PRs

For anything bigger than a single-file change, open an issue first so we
can agree on the shape before you spend time on it. New merging methods,
new subsystems, changes to the CLI or REST API all fall here.

## Claiming an issue

Comment on the issue before you start so we don't duplicate effort.
Issues tagged `good first issue` are self-contained.

## Adding an SE task

Task metadata lives in `mergese_tasks.py`. To add one:

1. Register the input format, metric, and default label columns.
2. Drop a small sample CSV under `data/benchmarks/` and add it to
   `data/benchmarks/index.json`.
3. Extend `tests/test_tasks_and_heads.py` with a load-and-merge test.

## License

Apache-2.0. Contributions are accepted under the same license.
