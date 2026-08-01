---
name: ponytail
description: Minimalism gate for Wakil-G. Run before adding any dependency, abstraction, service, table, event, queue, framework, new file category, or architectural mechanism. Prefer deletion, reuse, extension, or the smallest local change.
---

# Ponytail Gate

Run this gate whenever you (or an engineer) are about to introduce:
- a new package / library / pip dependency
- a new service, process, or container
- a new database table, column type, or index
- a new event type, queue, or async mechanism
- a new abstraction layer, base class, or interface
- a new file category (e.g., a new module folder, new config kind)
- any new architectural mechanism not already present in `SYSTEM_DESIGN.md`

## Gate Questions — answer all four in order

**1. Can this be deleted instead?**
Is the thing prompting this addition actually needed? Is there dead code, a redundant path, or an over-engineered component that should be removed rather than added to?

**2. Can something that already exists handle this?**
Read the relevant code. Does a current table, function, service, or pattern already cover the requirement — possibly with a small change?

**3. Can the existing thing be extended rather than replaced or supplemented?**
A new column beats a new table. A new field on an existing event beats a new event type. A predicate argument beats a new code path.

**4. What is the absolute smallest local change that satisfies the requirement?**
Write it down. If it is larger than "add one column" or "add one function," ask whether the requirement is scoped correctly.

## Decision

| All four answered honestly | Proceed? |
|---|---|
| Deletion or reuse satisfies it | No addition. Do the simpler thing. |
| Extension satisfies it | Extend. Do not add. |
| Nothing existing can handle it | Addition is permitted. Document *why* in the PR description. |

If the addition is permitted, it must be the **minimum viable version** — no speculative fields, no future-proofing, no optional hooks.

## Wakil-G Specific Notes

The authority store schema (bitemporal tables, lifecycle_effect, source_publication, bs_ad_calendar) is fixed by §2 and §4 of `SYSTEM_DESIGN.md`. New tables require an ADR and Prakash's approval — the gate does not unblock schema additions on its own.

The eval and stress harness (`make eval`, `make eval-gates`, `make stress`) must remain runnable after any addition. Do not add a dependency that breaks the test/eval surface.

New pip dependencies must not conflict with the adversarial-screening or ingestion pipeline's security posture — check before adding.

## Output Format

When running Ponytail, report:
```
Ponytail Gate — [thing being considered]

1. Delete instead? [Yes / No — reason]
2. Existing handles it? [Yes / No — what and how]
3. Extend existing? [Yes / No — how]
4. Smallest local change: [description]

Decision: [BLOCKED — use X instead | PERMITTED — minimum viable version is Y]
```
