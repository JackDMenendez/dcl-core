# notes/

Working theoretical / scratchpad notes that the package or its docs
build on. Notes are durable -- they survive package revisions -- and
are the natural home for material that is too long-form for a
docstring but too detailed (or too speculative) to promote to
`docs/design/`.

## Conventions

- One topic per file. Filenames are short and semantic
  (`born_rule_uniqueness.md`, not `notes-2026-04-01.md`).
- Markdown, not LaTeX. Notes are read in the editor; if a note grows
  enough mathematics to warrant a render, promote it to a design doc
  in `docs/design/` instead of upgrading the note format.
- Short header at the top: title, one-line purpose, status (DRAFT /
  STABLE / SUPERSEDED). When a note is superseded, leave it in place
  with a pointer to the replacement -- the project history is part of
  the project.
- Reference docs by relative path (`docs/design/02_remainder_strategy.md`),
  not by section number.

## Distinction from `docs/design/`

| | `notes/` | `docs/design/` |
|---|---|---|
| Audience | Maintainer (you, future-you) | Maintainers + new contributors |
| Polish | Rough, exploratory | Polished, durable |
| Lifecycle | Can be deleted or superseded freely | Survives across many releases |
| Example | "Trying out a complex-residual variant; here's why I think it might be wrong" | "We use a real-residual Bresenham; see issue #14 for the empirical justification" |

A note that has stabilised gets promoted to a design doc. A design
doc that turns out to be wrong gets superseded with a pointer to its
replacement (do not delete -- the history is part of the record).

## Examples

`example_note.md` shows the recommended structure -- replace it with
your own once the project has real content to capture.
