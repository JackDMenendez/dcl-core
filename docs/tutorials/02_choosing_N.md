# 02 -- Choosing `n_units` for your problem

`n_units` is the session's probability budget: the number of integer
tokens the session distributes across lattice sites. Larger `n_units`
gives finer spatial resolution of the distribution but costs more
memory and (for some operations) more time.

This tutorial walks through how to pick it for a specific target.

## The trade-off, in one paragraph

- **`epsilon_P = 1 / n_units`** is the smallest probability the
  framework can resolve.
- A bound state with characteristic radius `R` distributed over `~R^3`
  lattice sites needs `n_units` >> `R^3` to resolve sub-site detail;
  otherwise the tokens are too few to populate the distribution
  faithfully.
- Above the empirical drift floor measured by `exp_00_hop_drift`,
  discrete dynamics is observationally identical to continuous; below
  it, rounding artefacts will dominate.

## Calibration recipe

1. **Pick the physical target.** "Reproduce hydrogen Bohr radius to
   3 significant figures." Quantifies what "correct" means.
2. **Run `exp_00_hop_drift`** on the lattice shape you intend to use.
   It reports the per-tick analytical drift; `n_units` must be well
   above this for the framework's discreteness to be the binding
   constraint (not the hop's residual non-unitarity).
3. **Pick a starting `n_units`** at the lower end of the plausible
   range (say, `10^6` for a 16^3 lattice). Run the physical experiment.
4. **Double `n_units` repeatedly** until the headline number stops
   changing within your target precision. Record the smallest
   `n_units` that satisfies the target -- that's the "right" value
   for that physical setup.
5. **Sanity-check at the next-larger size**: refine again and confirm
   no further drift.

## Plausible orders of magnitude

| `n_units` | Cost | Typical use |
|---|---|---|
| `10^3 - 10^4` | trivial | Pedagogy / cellular-automaton coarse |
| `10^6 - 10^8` | seconds-to-minutes per tick on CPU | Quantitative reproductions on small lattices |
| `10^9 - 10^11` | needs GPU | Production runs on 64^3 - 128^3 lattices |
| `10^{12}+` | painful | Likely the regime where epsilon_P stops mattering |

For most physics questions, the framework's value comes from
**N being finite**, not from N being enormous. If you find yourself
needing `n_units > 10^{12}`, you have probably picked the wrong
question -- the dynamics is dominated by continuous-limit features
that the v1.0 (continuous-amplitude) core handles natively.

## What to record

For every physics experiment that uses this core, the experiment's
companion `.md` doc must record:

- `n_units` value used
- `lattice.shape`
- `epsilon_P = 1 / n_units`
- whether smaller `n_units` was tested and what changed
- the smallest `n_units` that satisfied the target precision

This is how a future paper that depends on this library can argue
that its quantitative claims aren't accidents of a fine-tuned
budget.
