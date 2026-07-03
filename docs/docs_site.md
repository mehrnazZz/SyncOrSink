# Documentation Site

SyncOrSink uses MkDocs for the public documentation site.

Public site:

```text
https://mehrnazzz.github.io/SyncOrSink/
```

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

Deployment is handled by `.github/workflows/docs.yml` on pushes to `main`.
In repository settings, GitHub Pages should use **GitHub Actions** as the source.
