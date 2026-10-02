# text2audiobook (production)

Fire-and-forget CLI for converting text sources (EPUB, PDF, Markdown, HTML, URL) to M4B audiobooks.

Primary documentation: [README.md](../README.md).

## Quick start

```bash
uv sync
uv run text2audiobook
# or
uv run text2audiobook --url https://retrochronic.com
```

Drop `.epub`, `.pdf`, `.md`, `.html`, or `.url` files into `data/input/`. A `.url` file is a single http(s) URL on the first line (optional title on line 2). Fetched pages are split on `h1`/`h2` headings after stripping site chrome.
