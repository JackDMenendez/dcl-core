# docs/

Documentation lives in three buckets, separated by audience and lifecycle:

| Directory | Audience | Lifecycle |
|---|---|---|
| `tutorials/` | New users / Claude on first contact | Few, stable, hand-curated |
| `reference/` | Returning users looking up API | One per public module / class |
| `design/` | Maintainers (incl. future-you) | Rationale; rarely deleted, sometimes superseded |

## Building docs (optional)

The `[docs]` extra installs `mkdocs` + `mkdocs-material`:

```sh
pip install -e .[docs]
mkdocs serve         # live preview on http://localhost:8000
mkdocs build         # writes to site/
```

`make docs` runs the build target. A `mkdocs.yml` config is not
provided by the template -- add one at the repo root once the docs
have stabilised. For early-stage work, reading the Markdown files
directly in the editor is sufficient; the build step is for
publication.

## Cross-references

When code refers to a design doc or a tutorial, use a path relative
to the repo root, e.g.

```python
"""...

See `docs/design/02_remainder_strategy.md` for the rationale.
"""
```

Editors render this as a clickable link; CI lint rules in future can
verify that the linked path exists.

## What does NOT go here

- Working theoretical scratchpad notes -- those go in `notes/`. A note
  is durable but rough; a doc is durable and polished.
- Release-by-release change logs -- those go in `release_notes/`.
- Section-of-a-paper content -- belongs in a separate paper repo.
