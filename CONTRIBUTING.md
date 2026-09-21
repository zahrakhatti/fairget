# Contributing to fairget

Contributions that improve correctness, documentation, testing, or the public
API are welcome. Before proposing a larger change, open an issue or start a
Discussion so the scope can be agreed on first.

## Development setup

```bash
git clone https://github.com/zahrakhatti/fairget.git
cd fairget
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Pull requests

- Keep each pull request focused on one problem.
- Add or update tests when behavior changes.
- Update the README when the public API or installation steps change.
- Run the full test suite before submitting.
- Explain any modeling assumptions and report relevant fairness and utility
  effects for changes to constraints, metrics, or examples.
- Use synthetic or publicly redistributable data in tests and examples. Do not
  commit private, sensitive, or restricted datasets.

Bug reports are most useful when they include a minimal example, the `fairget`
version or commit, the Python and PyTorch versions, and the complete error
message.
