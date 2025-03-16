# CLAUDE.md - Solitaire Project Guide

## Build & Test Commands

### Python

From the `py` directory:
- Run tests: `uv run pytest`
- Run single test: `uv run pytest test_klondike.py::test_name`
- Type check: `uv run pyright`
- Linting: `uv run ruff check .`
- Format code: `uv run ruff format .`

### Rust

From the `rs` directory:
- Run tests: `cargo test`
- Run single test: `cargo test test_name`
- Run main: `cargo run`
- Type check: `cargo check`
- Linting: `cargo clippy`
- Format code: `cargo fmt`

## Code Style Guidelines

### Python

- Python 3.12+ compatible code
- Line length: 120 characters max
- Use type hints for all functions and variables
- Use dataclasses for structured data
- Use Enums for categorical data
- Imports: standard library first, then third-party, then local
- Follow PEP 8 naming conventions
  - snake_case for variables, functions, methods
  - PascalCase for classes and types
  - UPPER_CASE for constants
- Error handling: use raised exceptions with descriptive messages
- Testing: use pytest fixtures and assertions
- Comments: use docstrings for public API
