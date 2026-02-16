# AIMO 3 - Technical Requirements & Notation

## LaTeX Notation Conventions

### Packages Used
- `amsmath`
- `amssymb`
- `amsthm`
- No other packages assumed

### Mathematical Notation

#### Basic Notation
| Notation | Meaning |
|----------|---------|
| `$n!$` | Factorial (0! = 1) |
| `$\|x\|$` | Absolute value |
| `$\lfloor x \rfloor$` | Floor function (greatest integer <= x) |
| `$\lceil x \rceil$` | Ceiling function (smallest integer >= x) |
| `$\{x\}$` | Fractional part (x - floor(x)) |
| `$\binom{n}{k}$` | Binomial coefficient |
| `$\log$` | Natural logarithm (unless base specified) |
| `$:=$` | Defined to be equal to |

#### Sets
| Notation | Set |
|----------|-----|
| `$\mathbb{N}$` | Positive integers (>0) |
| `$\mathbb{Z}$` | Integers |
| `$\mathbb{Q}$` | Rational numbers |
| `$\mathbb{R}$` | Real numbers |
| `$\mathbb{Z}/n\mathbb{Z}$` | Integers modulo n |

#### Sequences & Operations
- `$\ldots$`, `$\cdots$`, `$\dots$` for finite/infinite sequences
- Multiplication: adjacency `$xy$`, `$\cdot$` (`$x \cdot y$`), or `$\times$` (`$x \times y$`)
- Overline `$\overline{abc}$` = integer formed by digits a,b,c (e.g., $\overline{abc} = 100a + 10b + c$)

#### Modular Arithmetic
- Remainder when X divided by Y: answer is unique r where 0 <= r < Y and Y divides (X - r)
- Example: `2^50 mod 1001 = 559`

### Geometry Conventions
- Triangle ABC has sides a, b, c (opposite to vertices A, B, C)
- Altitudes: "A-altitude" or "altitude from A"
- Angles may be in degrees or radians (radians if no unit specified)
- Trapezium = quadrilateral with at least one pair of parallel opposite sides
- Taxonomy is Bourbakist: equilateral triangles are isosceles, squares are rectangles

### Environments Used
- `align`, `align*`
- `equation`, `equation*`
- `gather`, `gather*`
- `cases` (from amsmath)

## Problem Characteristics

### Domain Distribution
1. **Algebra** - equations, inequalities, polynomials, sequences
2. **Combinatorics** - counting, probability, graph theory
3. **Geometry** - plane geometry, coordinate geometry, trigonometry
4. **Number Theory** - divisibility, primes, modular arithmetic

### Difficulty Range
- Lower bound: National Mathematical Olympiad level
- Upper bound: International Mathematical Olympiad (IMO) level
- Some problems slightly easier/harder than this range

### Design Philosophy
- All problems are original (zero contamination risk)
- Designed to be "AI hard" - tested against current open LLMs
- Require genuine mathematical reasoning
- Answer-only testing designed to be robust
- No diagrams (geometry problems are text-only)

## Code Environment

### Available Resources
- H100 GPUs (AIMO 3 exclusive, no internet)
- Standard Kaggle accelerators
- Pre-trained models allowed (publicly available)
- External datasets allowed (publicly available)

### Restrictions
- No internet access during inference
- Must use provided evaluation API
- 5-hour GPU runtime limit / 9-hour CPU limit

## Partner Resources (Application Required)

### Fields Model Initiative
- Up to 128 H100 GPUs for fine-tuning
- Supported by LLMC (NII Tokyo) and Benchmarks+Baselines (Vienna)
- Applications open December 2025

### Thinking Machines (Tinker)
- Up to $400 in API credits
- Abstracts away engineering complexity
- Good for mathematicians less familiar with ML engineering

## Sources
- [Kaggle Competition Rules](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/rules)
- [AIMO Reference Problems PDF](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/data)
