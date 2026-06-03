# Color structure: geometric RGB vs. the C^3 color factor

**Status:** DRAFT
**Purpose:** Record the *corrected* picture of how the lattice's
geometric color vectors {V_r, V_g, V_b} relate to SU(3) color, so
proton-internals work in `core3d` starts from Paper II's result and
not from the naive (wrong) "RGB vectors ARE color charge" reading.
**Cited by:** (none yet -- will be cited by `core3d` color work and
any `docs/design/` doc on proton internals)

---

## Setup

The current engines (`core`, `core3d`) carry a per-site amplitude
of `C^2` only: the Dirac chirality spinor `(psi_R, psi_L)` on the
bipartite RGB/CMY sublattices (see
`src/dcl_core/core/CausalSession.py`, `src/dcl_core/core/OctahedralLattice.py`).
The six basis vectors are

    RGB:  V_r=(1,1,1)  V_g=(1,-1,-1)  V_b=(-1,1,-1)
    CMY:  V_c=-V_r     V_m=-V_g       V_y=-V_b

`CMY = -RGB` exactly; anticolor IS the negation of color.

The natural temptation when moving toward proton internals is to
read {V_r, V_g, V_b} as the three QCD color charges directly, a
proton as three quarks living one-per-vector, and the color-singlet
("white") condition as some vanishing of a vector sum. **This
reading is wrong**, and Paper II proves why. This note records the
correct picture.

## Argument

### 1. The geometric RGB vectors give only Z_3, not SU(3)

Paper II's `src/utilities/automorphism_rgb_su3.py` (sympy-verified,
audit status PASS) establishes:

- The 6 permutations of {V_1, V_2, V_3} embed in U(3) as 3x3
  unitaries on the C^3 weight space.
- Only the 3 *even* permutations have `det = +1` and lie in SU(3);
  they form the cyclic group **Z_3 = A_3**. The 3 transpositions
  have `det = -1` (in U(3) but not SU(3)).
- Z_3 is abelian and finite. It does **not** generate the
  non-abelian SU(3) (dim 8) by closure, commutators, or
  exponentiation. An abelian group cannot produce a non-abelian
  algebra.

Verbatim conclusion: *"SU(3) is a structural ADDITION to the
framework, not a direct continuation of its discrete RGB symmetry."*

So the geometric color vectors supply only the **abelian Cartan
corner** of color.

### 2. Real color lives on a separate C^3 "color memory" factor

Paper II's resolution (`notes/su3_generation_from_colour_memory.md`,
PASS) extends the per-site amplitude by a tensor factor:

    C^2  ->  C^2 (x) C^3
    full SM:  C^12 = C^2 (chirality) (x) C^2 (weak isospin) (x) C^3 (colour)

The C^3 is a **color memory**: which of (V_1, V_2, V_3) the
wavefunction traversed most recently. The geometric vectors do not
*carry* color; their *traversal acts on* this internal color index.

Full su(3) (dim 8) is then **generated** by the color-memory tick
rule whenever the rule has real-symmetric "rotate-toward-|j>"
off-diagonal content -- robustly: even a dim-2 seed closes to dim 8
under Lie brackets. (Purely-diagonal rules give only the Cartan
dim-2; purely-imaginary-antisymmetric give one su(2) dim-3. These
are the degenerate corners.)

### 3. The geometry-color bridge

The geometric RGB symmetry does not vanish from the story -- it
**organizes** the color factor. The three color-memory unitaries are
tied together by the lattice's S_3 basis symmetry:

    U_j = P_{1->j} U_1 P_{1->j}^{-1}

So {V_r, V_g, V_b} become central to proton internals as **(a)** the
spatial directions whose traversal drives the color-memory rotation,
and **(b)** the S_3 symmetry relating the three color-memory
operators -- *not* as the color charges themselves. The discrete RGB
Z_3 of step 1 is exactly the abelian Cartan part of the full su(3)
generated in step 2.

### 4. Consequences for `dcl_core`

- Proton internals require **two distinct code objects**, not one:
  the spatial hop direction `V_j` (already present as
  `RGB_VECTORS`), and a separate per-site `C^3` color factor with
  its own su(3) action carried by the color-memory tick rule.
- This is the point at which Paper II stops being "pure symbolic, no
  engine dependency." Paper II proved the C^3 color algebra exists
  *symbolically*; implementing proton internals in `core3d` is
  exactly the act of making that algebra **run numerically**. That
  is a new bridge between the two repos that does not exist today.
- The color factor is part of Paper II's larger left-right-symmetric
  centralizer (dim-71: su(6) + su(6) + u(1), with the SM's dim-18 as
  a factor-product projection). Proton internals and the
  complex-`BresenhamResidual.carry` / "extra generators" hypothesis
  may be touching the same object from two sides.

## Open questions

1. **Color-singlet / baryon condition.** What IS "colorless" on the
   lattice? It lives in the internal C^3 color space, *not* position
   space. The position-space combination `V_r + V_g + V_b = (1,1,-1)`
   is the Cartan/Z_3 shadow, not the singlet condition. The actual
   color-singlet projector on C^3 is still to be defined (CONJECTURED).
2. **Which color-memory tick rule?** Paper II shows many candidates
   close to the same su(3); does proton dynamics select one, or is
   the non-uniqueness physical?
3. **Does the color factor have a mass-analogue amplitude**, the way
   chirality mixing becomes mass via clock density? (Flagged in
   Paper II's su3 note, upstream-flow tags.)
4. **Relationship to the dim-71 centralizer / carry-extras.** Is the
   complex carry the same structure as the color-or-wider internal
   factor? (See memory: complex-carry hypothesis.)

## Provenance tags (for any future Correspondence registry)

| Correspondence | Provenance |
|---|---|
| `V_j` permutation symmetry -> Z_3 subset SU(3) | STANDARD (Paper II PASS) |
| `C^3` color-memory index -> full su(3) | established-in-Paper-II (PASS, candidate-dependent) |
| traversal `V_j` -> color-memory rotation `U_j` | NOVEL (the geometry<->color bridge) |
| color-singlet / baryon condition in `C^3` | CONJECTURED |

## Pointers

- Related notes: `notes/bresenham_residual_design.md`;
  memory `complex_carry_hypothesis.md`.
- Paper II (`external/dcl-paper-02-sm-derivation`, i.e.
  `C:\dev\dcl-paper-02-sm-derivation`):
  - `src/utilities/automorphism_rgb_su3.py` -- RGB -> Z_3 (step 1).
  - `notes/su3_generation_from_colour_memory.md` and
    `src/utilities/su3_generation_from_colour_memory.py` -- C^3
    memory -> full su(3) (step 2).
  - `notes/su3_branch_consistency.md` -- Branch A SU(3) as global
    symmetry vs. dynamically generated.
  - `README.md` -- the dim-18 conjecture and dim-71 centralizer.
- Related code: `src/dcl_core/core/OctahedralLattice.py`
  (`RGB_VECTORS`/`CMY_VECTORS`), `src/dcl_core/core/CausalSession.py`
  (the `C^2` spinor this would extend).
