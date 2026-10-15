# Contributing to quantlab

Thank you for your interest in quantlab. Bug reports, questions, documentation fixes and code
changes are all welcome. This page explains how to set up a development environment and what
we look for in a pull request.

## Reporting a bug or asking a question

Open an issue at https://github.com/ZhaorongDai/quantlab/issues. For a bug, include the
command or code you ran, the full error message, your operating system and your Python
version. A small example that reproduces the problem on synthetic data is the most helpful
thing you can provide. Never paste API keys, WRDS passwords or other credentials into an
issue.

## Setting up a development environment

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
cd quantlab
uv sync --all-extras
uv run pytest
```

The test suite runs offline and needs no credentials. See the
[installation guide](docs/getting-started/installation.md) for details, including the one test
module that needs a real CRSP store.

## Making a change

Create a branch for your change and keep each pull request focused on one topic. Before you
open the pull request, run the tests and make sure they pass. If you fix a bug, add a test
that fails without your fix. If you add a feature, add tests for it and update the relevant
page under `docs/`.

The [developer guide](docs/developer-guide/extending.md) shows how to add the most common
kinds of extension: a data source, a dataset, a storage backend, a factor, a model head and a
backtest rule. Each stage talks to the next only through the panel format described in
[Concepts](docs/user-guide/concepts.md), so a new component should accept and return panels
indexed by `timestamp` and `symbol`.

## Code conventions

Keep configuration in the config dataclasses in `quantlab/base/config.py` rather than in
hard-coded values, so that runs stay reproducible. Read credentials only from environment
variables, never from code or configuration files. Abstract base classes live in
`quantlab/base/`, and their concrete implementations live with the code that uses them.

## Docstrings and comments

Every module, class and function has a docstring in the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format. Start with a
one-sentence summary, then a short paragraph if the behaviour needs explaining, then the
`Parameters`, `Returns` and `Raises` sections, and an `Examples` section where a short
example helps. Examples must show real output; if an example cannot run offline, show it as a
plain code block without output.

Write for a reader who knows Python and basic finance but not this project. Explain a
domain term the first time a module uses it, prefer plain sentences to long bullet lists,
and describe what the code does now rather than how it got there. Comments should explain
why the code does something, not repeat what it does.

## Releasing

A release is built and published by `.github/workflows/release.yml`, never from a working
tree: a local build packages whatever sits on disk, untracked files included. To release,
set `version` in `pyproject.toml`, commit, then tag and push:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow checks that the tag matches the version, builds from the tag's checkout,
runs `twine check`, imports every module from the installed wheel, and publishes through
PyPI trusted publishing. Trusted publishing must be set up once on PyPI for the
`ZhaorongDai/quantlab` repository, workflow `release.yml` and environment `pypi`. A
version on PyPI cannot be replaced, so a manual run (`workflow_dispatch`) builds and checks
without publishing.
