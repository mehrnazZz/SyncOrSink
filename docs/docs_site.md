# Documentation Site

SyncOrSink uses MkDocs for the public documentation site.

Install docs dependencies:

```bash
pip install -e ".[docs]"
```

Preview locally:

```bash
mkdocs serve
```

Build with the same strictness as CI:

```bash
mkdocs build --strict
```

The generated `site/` directory is ignored by git. Commit source Markdown,
`mkdocs.yml`, and generated leaderboard tables, not built HTML.
