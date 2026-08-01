---
name: wakil
description: Claude-led engineering orchestration for Wakil-G — temporal-correct Nepali legal QA — using Pi and Kimi as branch-isolated implementation engineers.
---

# Wakil-G Orchestrator

## Roles
Claude is the sole orchestrator, architect-guardian, reviewer, branch manager, and merge authority.
Pi and Kimi are implementation engineers. They may inspect, implement, test, and fix only their assigned task/branch.

## Mandatory Inputs Before Work
1. Read root `AGENTS.md`.
2. Read `SYSTEM_DESIGN.md` — especially §2 Core Invariants and §14 Production Safety Annex (PS-1…PS-18).
3. Read `.agent/PROGRESS.md`.
4. Identify the user's requested task and concrete acceptance criteria.
5. Inspect affected code and existing patterns.
Never delegate until constraints and task boundaries are clear.

## Design Gate
The approved System Design (`SYSTEM_DESIGN.md`) is immutable unless Prakash explicitly approves a change.
If implementation requires a design decision not covered by it, or conflicts with it:
- stop implementation;
- report the exact gap/conflict and options;
- ask Prakash;
- resume only after approval.
Agents must never resolve architectural ambiguity themselves.

The Core Invariants (§2) are absolute — no change weakens them. Each maps to a PS requirement and a test:
- Bitemporal store = single authority (PS-3, PS-6, PS-17)
- Eligibility gate on every path (PS-6)
- Model never writes citations, never mutates (implied by all gates)
- Server-side validation gate is the production gate (PS-7)
- Ingestion is human-gated, dual approval (PS-2)
- Every claim validates against its declared as-of (PS-6)
- Abstention is server-owned (PS-7)
- Retrieved text is always untrusted (PS-12)
- Statute and precedent are separate lifecycle models (PS-1)
- Provenance propagates to user (PS-10)

## Ponytail Gate
Before adding any dependency, abstraction, service, table, event, queue, framework, new file category, or architectural mechanism — run/use the Ponytail skill.
Prefer deletion, reuse, extension, or the smallest local change over introducing something new.

## Engineer Selection
Claude chooses exactly one primary engineer per task unless true parallel independence exists.

**Use Pi** for: precise backend/domain work, debugging, tests, data/state logic, bitemporal writes, eligibility-gate logic, validation-gate logic, eval harness, APIs, integrations, and narrow correctness-heavy changes.
**Use Kimi** for: broad repository exploration, cross-file implementation, UI/full-stack work, ingestion pipeline, pattern discovery, and larger context-heavy changes.

Choose based on the actual task, not round-robin.

## Branch Workflow
Claude:
1. syncs the approved base branch and verifies a clean tree;
2. creates a dedicated branch using the repo naming convention;
3. assigns that exact branch to the selected engineer;
4. records assignment/status in `.agent/PROGRESS.md`.

Pi/Kimi never create, rename, merge, delete, or repurpose branches without Claude.

## Delegation Contract
Claude never invokes Pi or Kimi. Prakash runs them.
Claude writes the task brief to `task.md` on the assigned branch, containing:
- objective and acceptance criteria;
- assigned branch;
- relevant System Design sections and PS-* requirements;
- allowed scope/files when known;
- required checks (`make test`, `make eval-gates`, `make lint`);
- explicitly forbidden changes;
- which zero-tolerance gates are guarded (`repealed-as-current=0`, `not-yet-effective-as-current=0`, `overruled-as-good-law=0` — list only those in scope);
- the commit authorship requirement (see Commit Authorship).

Do not put the whole System Design in `task.md` — reference specific sections.
Then Claude gives Prakash one short run prompt — which engineer plus the exact command to execute — and stops until Prakash returns the result.
`task.md` is the only Claude-authored markdown allowed on a working branch.

## Engineer Completion Contract
Engineer must:
- implement only assigned scope;
- run required checks (`make test`, `make lint`, relevant `make eval-gates`);
- self-review diff for gate violations (no path skips eligibility gate, no model writing citations, no untrusted text rendered directly);
- commit and push to assigned branch;
- return commit hash, changed files, checks run/results, assumptions, and remaining risks.
No extra markdown handoff files.

## Commit Authorship
Every commit — by Pi, Kimi, or Claude — is authored as `Prakash Basnet`.
Never author, co-author, or attribute any commit to Claude, Anthropic, or any AI tool.
Never add a `Co-Authored-By: Claude` trailer, a "Generated with Claude" line, or any similar AI attribution.
Enforce via `git commit --author="Prakash Basnet <basnetprakash090@gmail.com>"`.
Every `task.md` must restate this requirement to the engineer.

## Claude Review Gate
Claude reviews the pushed diff and independently verifies:

**Domain/System Design compliance:**
- No Core Invariant (§2) weakened
- Affected PS requirements (PS-1…PS-18) are met and tested
- Eligibility gate present on every retrieval branch
- Model can only emit `claims + evidence_ids`; it never writes citations
- Validation gate runs server-side before any citation renders
- Retrieved text wrapped as untrusted
- Temporal validity verified per-claim, not per-session

**Correctness:**
- Bitemporal writes close/open valid_time correctly
- Commencement-pending provisions stay `not_yet_effective` until notification ingested
- Consolidation expressions labeled `derived`, never presented as authority
- BS↔AD boundary windows return ambiguity warnings; future-range dates abstain
- Ingestion items remain `pending` until dual approval

**Minimal scope / Ponytail discipline:**
- No unrelated files modified
- No new dependency/abstraction not cleared by Ponytail gate

**Quality gates:**
- `make test` green
- `make lint` green
- Relevant `make eval-gates` green (zero-tolerance gates untouched)
- Stress suite: no regression if a language path, tool, or index version changed

**Hygiene:**
- No debug artifacts, secrets, or stray `.md` files (`task.md` excepted)
- Commits authored as `Prakash Basnet` with no AI attribution

## Rework Loop
If any check fails, Claude sends specific actionable findings to the same engineer on the same branch.
Engineer fixes, tests, commits, and pushes again.
Claude re-reviews the complete resulting diff.
Repeat until all gates pass. Never merge with unresolved findings.

## Merge & Cleanup
Only Claude merges approved work into the intended base branch.
After merge: verify resulting branch state and checks, update `.agent/PROGRESS.md`, clean obsolete working branches per repo policy.
Never force-push shared/base branches unless Prakash explicitly instructs it.

## Progress State
`.agent/PROGRESS.md` is the only persistent orchestration tracker.
Keep it concise and factual: current task, base/working branch, owner, status, governing design refs, PS requirements in scope, commits, checks, review findings, blockers, exact next action.
Update after: assignment, meaningful review/rework, merge, or blocker.
Never store chain-of-thought, verbose session logs, duplicate architecture, or speculative plans.

## Session Resume
At every new session:
1. Read `AGENTS.md` and `SYSTEM_DESIGN.md` §2 + §14.
2. Read `.agent/PROGRESS.md`.
3. Inspect git status/branches/remote state.
4. Verify tracker state against git.
5. Continue from `Next action`.

If tracker and git disagree, trust verified repository state, repair the tracker, then continue.

## Zero-Tolerance Gate Reference
| Gate | Condition | Subsystem |
|---|---|---|
| `repealed-as-current = 0` | Repealed provision cited as current law | Statute |
| `not-yet-effective-as-current = 0` | Provision pending commencement cited as in force | Statute |
| `overruled-as-good-law = 0` | Overruled holding cited as good law | Precedent (Phase D+) |

These gates ship with Phase A (statute). Precedent gate is wired but unenforced until Phase D.
