# Wakil-G — System Design

**Project:** Wakil-G — a temporal-correct legal question-answering system for Nepali law (statute + precedent).
**Status:** canonical architecture reference. Tech stack is swappable; the **Core Invariants** in §2 are not.

> Prime directive: **a loud refusal beats a quiet wrong answer.** Every design choice below exists to prevent the system from confidently citing law that is repealed, not-yet-in-force, overruled, void, or misdated.

---

## 1. Scope

Wakil-G answers legal questions grounded in authoritative Nepali sources (Acts, Rules, Rajpatra/Gazette instruments, Supreme Court नजिर), each answer bound to an explicit point in time (as-of), with citations that resolve to authoritative instruments — not to model output and not to a system-generated consolidation presented as authority.

Out of scope (stated so it is never silently assumed in scope): adjudicating constitutional validity or vires, giving legal advice, and treating the system's `in_force` flag as a statement about a provision's legal validity.

---

## 2. Core Invariants — the parts that never change

These define Wakil-G independent of any language, database, or model choice. A change that violates one of these is an architecture change, not an implementation detail.

1. **The bitemporal store is the single authority.** Search indexes, embeddings, and materialized consolidations are *derivatives*, never sources of truth. Every citation is revalidated against the authority store before it ships.
2. **Deterministic eligibility gate runs pre-retrieval, on every branch.** Temporal validity, status, jurisdiction, source eligibility, and ACL are filtered *before* retrieval, with the same predicate on every path.
3. **The model never writes citations and never holds mutation power.** It emits `claims + evidence_ids` only. It sees read-only resolves and untrusted context — nothing more.
4. **A server-side validation gate is the production gate.** It resolves evidence IDs, verifies span hashes, verifies temporal validity at the answer's as-of, checks claim support, and only then renders citations from canonical metadata. No answer bypasses it.
5. **Ingestion of legal state is human-gated with dual approval.** No source becomes searchable legal state without a Gatekeeper + second approver. Rejected items stay `pending` and are never treated as law.
6. **Every query resolves an explicit as-of; every claim validates against its declared as-of.** (Not one as-of per answer — see §7.2. Comparative/history queries carry a *set* of as-of points, one per claim.)
7. **Abstention is server-owned.** The model's self-abstention is an advisory prior only. The authoritative decision to abstain is the validation gate's claim-support + threshold check.
8. **All retrieved text is untrusted.** Statutory text is an injection surface; it is wrapped as untrusted and canary-monitored. Citations are rendered from canonical metadata, never copied from model output.
9. **Statute and precedent are separate lifecycle models with separate zero-tolerance eval gates.** `repealed-as-current = 0` for statute; `overruled-as-good-law = 0` for precedent. One model does not serve both.
10. **Provenance propagates to the user.** Source kind, OCR confidence, translation status, and "derived consolidation" markers travel with every citation into the UI.

---

## 3. High-level architecture

```
                         ┌───────────────────────────────────────┐
                         │         AUTHORITY (PostgreSQL)         │
                         │  bitemporal statute + precedent +      │
                         │  lifecycle events + sources + BS↔AD    │
                         └───────────────────────────────────────┘
                             ▲ (write, human-gated)   │ (revalidate, read)
   SIDE A — INGESTION        │                        │        SIDE B — QUERY
   (offline, human-gated)    │                        ▼        (online, deterministic-first)
 ┌──────────────────────┐    │            ┌──────────────────────────┐
 │ sources → raw store  │    │            │ query → guardrail →      │
 │ → adversarial screen │    │            │ understanding → AS-OF    │
 │ → structure + life-  │────┘            │ → ELIGIBILITY GATE       │
 │   cycle extraction   │                 │ → retrieval → fusion     │
 │ → HUMAN VALIDATION   │                 │ → context (UNTRUSTED)    │
 │ → bitemporal write   │                 │ → router → generation    │
 │ → re-embed affected  │───────┐         │   (claims+evidence_ids)  │
 └──────────────────────┘       │         │ → VALIDATION GATE        │
                                ▼         │ → output guardrail       │
                    ┌──────────────────────┐  └──────────────────────┘
                    │ SEARCH DERIVATIVES   │             ▲
                    │ (BM25 + kNN indexes) │─────────────┘ (retrieve; then
                    └──────────────────────┘                revalidate vs authority)

 CROSS-CUTTING: Stress-test layer (3 surfaces) · Eval layer (sliced golden set) ·
                Observability (OTel, redacted) · Degraded-mode ladder · Feedback loop
```

Two planes share one authority store. Ingestion writes to it under human gate; query reads from derivatives but **revalidates against authority before any citation renders**.

---

## 4. Data model

The authority store is bitemporal on two independent axes:

- **legal_valid_time** — when a provision is the law (moves on amendment/repeal/commencement; can be retroactive).
- **transaction_time** — when the system recorded it (moves on correction). These axes diverge by design; retroactive amendments have `valid_time` earlier than `transaction_time`.

Core entities:

- **work** — a statute/Rule/instrument as an identity (`work_type` ∈ Constitution / Act / Rule / Directive / Notification, giving norm level).
- **component** — परिच्छेद / भाग / दफा / उपदफा / खण्ड / अनुसूची / proviso / स्पष्टीकरण, addressable by URI. Provisos and स्पष्टीकरण are components *of* their operative clause and never chunked away from it.
- **expression** — a materialized as-amended reading of a component at a valid-time. **Derived, non-authoritative** (see §7.4).
- **source_publication** — a physical source with `kind` ∈ official_original / amending_instrument / verified_internal_consolidation / official_copy_unverified / derived_verified, plus OCR confidence. Kinds are never silently merged.
- **lifecycle_effect** — per-component events: amend / repeal / commence / expiry / suspend / correct / **declared_invalid** (from a court source). Carries targets, effective date, replacement text, source span, and — where applicable — a **commencement_dependency** (§7.3) and an **enabling_power** link (§7.5).
- **precedent** + **precedent_relation** — separate subsystem, §6.
- **bs_ad_calendar** — versioned canonical calendar table, §7.6.

---

## 5. Ingestion plane (Side A)

Offline, human-gated. Flow: **sources → immutable raw store** (write-once, SHA-256) → **adversarial screening** (injection/hidden-instruction scan, OCR corruption scoring, source-kind classification) → **structure pipeline** (Unicode NFC → digit fold → Indic normalization → structural parser with URIs → lifecycle extraction, *proposal only*) → **human validation** (Gatekeeper verifies source/target/operation/effective-date/resulting-text/span; second approver; dual approval mandatory for lifecycle) → **bitemporal write** (close/open valid-time or transaction-time as appropriate; materialize expression; re-embed only affected components; temporal regression tests; publish) → **search derivatives** (versioned indexes/aliases; authority stays in Postgres).

Rejected items remain `pending` and are never searchable as legal state.

---

## 6. Precedent subsystem (statute model does NOT cover this)

Case law does not have a statutory lifecycle. A नजिर is not amended or repealed; it is overruled, reversed, distinguished, affirmed, or questioned — a *relation between works*, not an event on one work.

**Model:**

- `precedent_relation(source_case, target_holding, relation_type ∈ {overrules, reverses, distinguishes, affirms, questions}, bench_strength, valid_time, source_span)` — human-extracted, same dual-approval gate as lifecycle.
- **Target is a holding/proposition, not the whole case.** A case overruled on point X remains good law on point Y. Status is derived at the holding level, never as a case-level boolean.
- **Bench competence gates effectiveness.** An overruling by a smaller/incompetent bench than the target is *per incuriam* and legally ineffective; the derivation predicate is "verified `overrules`/`reverses` **from a competent-or-larger bench**," not "any such relation."
- **Relations are themselves temporal.** An overruling can later be reversed; relations carry `valid_time`, and "good law" is derived from the relation set **as-of the answer date**.

**Eval gate:** `overruled-as-good-law = 0` (holding-level). Case law does not ship until this subsystem exists (Phase D). Until then, Supreme Court is ingested but not answered from.

---

## 7. The correctness core

### 7.1 Deterministic eligibility gate (pre-retrieval, non-negotiable)
Predicate applied identically on every retrieval branch:
`legal_valid_time @> as_of AND status='in_force' AND jurisdiction AND source_eligible AND acl`.
OpenSearch hits are revalidated against Postgres before any citation renders.

### 7.2 As-of, per-claim
The session resolves a default as-of once at creation and passes it to every subquery and tool call; tool APIs reject calls with no as-of. But the enforced invariant is **per-claim**: each claim carries its declared as-of and validates against *that*. This permits diachronic queries ("how did दफा X change 2070→2080") to legitimately span multiple as-of points, while the gate still rejects any claim whose evidence doesn't match the as-of it declared.

### 7.3 Commencement dependency
For provisions commencing *"राजपत्रमा सूचना प्रकाशन गरी तोकिएको मितिदेखि"*: the parser flags the pattern at ingestion and forces `status='not_yet_effective'` with an explicit **"commencement pending notification"** display state (a third state, never silently in-force, never silent). A `commencement_dependency(pending_instrument_type='gazette_notification')` links the provision to the future instrument. When the notification is ingested and verified, it materializes valid-time for only the enumerated sections. Effective date is `NULL-pending`, never a fabricated date.

### 7.4 Consolidation is not authority
The materialized `expression` is a reading convenience, labeled **derived**. Citations resolve to the **authoritative chain**: base-Act source span + each amending instrument's source span — rendered from `source_publication` rows of kind `original`/`amending_instrument`, never from the expression. CI invariant: for every expression, regenerate from base + verified effects and diff against the stored expression over the **normalized** form (NFC, digit-fold, structural whitespace canonicalized, to avoid flaky false-fails). If it doesn't reconstruct, it doesn't publish.

### 7.5 Norm hierarchy & vires (recorded facts only)
Store what is knowable: norm level (`work_type`), competence subject list, an **`enabling_power`** link from subordinate legislation to its parent enabling provision (so orphaned Rules are flagged when the enabling section is repealed), and court invalidity rulings as `declared_invalid` lifecycle events. Do **not** model constitutional validity as DB truth — that is a judicial determination. Product states plainly: "`in_force` here = recorded as in force, ≠ a validity opinion."

### 7.6 BS↔AD as canonical data
The official panchanga mapping is **versioned canonical data with provenance**, ingested like any other source — not a library call. A reviewed **disagreement table** flags month-boundary windows where library outputs diverge from the official mapping; any query date in a flagged window returns the date plus an ambiguity warning. Temporal validation tests sample boundary dates specifically. **Future dates beyond the published panchanga range refuse silent conversion** and abstain — a one-day error selects the wrong version, exactly the failure the bitemporal design exists to kill.

### 7.7 Server-side validation gate (the production gate)
Resolve evidence_ids → verify span hash → verify temporal validity at the claim's as-of → revalidate vs Postgres → exact-quote check → claim-support check → bounded regenerate (max N) → then fallback/abstain → render citations from canonical metadata. This gate, not the model, owns abstention and citation rendering.

---

## 8. Query plane (Side B)

`query → input guardrail (jailbreak/PII/rate+cost caps) → understanding (NFC, digit-fold, script/language ID confidence-scored, legal-reference parser, variants — original always preserved) → as-of resolution (+ BS↔AD) → eligibility gate → parallel retrieval over eligible set (exact URI · title · citation tokens · BM25 Nepali · multilingual vector · variants) → fusion (RRF) → rerank (benchmark-approved only) → context assembly (parents, definitions, cross-refs, provisos, schedules, temporal + pending-verification warnings; all wrapped UNTRUSTED) → router → generation → validation gate → output guardrail → user + trace`.

**Router:** ordinary query → bounded deterministic RAG. Multi-issue / cross-jurisdiction / explicit research → orchestrator with allowlisted read-only tools only, iteration/token/wall-clock caps, subqueries treated as untrusted, **no admin mutation tools in existence**, and the session as-of enforced on every hop.

---

## 9. Degraded-mode ladder (defined, chaos-drill tested)

- Model gateway down → retrieval-only extractive answers with citations.
- Reranker down → BM25-only path, flagged.
- OpenSearch degraded → Postgres exact-lookup fallback.
- Reviewer backlog → answers carry "amendment pending verification."
- **Authority (Postgres) down → no validated answers, period.** Never fall back to trusting index temporal metadata. Revalidation runs on a **synchronous or freshness-gated replica** (check replica applied-LSN against the latest lifecycle-write watermark; refuse to revalidate on a lagged replica — an async replica can certify a repealed provision as current). Signed span-hash cache permits re-rendering *already-validated* content (integrity ≠ temporal validity; never new validation). Primary + replica both down → loud refusal with status page.

---

## 10. Evaluation & stress layers

**Eval (from Phase 0 onward):** sliced golden set (language × query-type × temporal × jurisdiction), numeric target per slice. **Romanized Nepali is its own slice with its own target** (no orthographic standard → hardest input, likely largest real-user cohort; folding it into "mixed" masks it). LLM-judge is **triage only, κ ≥ 0.6 overall** as deploy-blocker, tracked per slice and rolling (a judge-model update that silently drops κ invalidates months of triage); slices below threshold get 100% human review. Zero-tolerance gates: `repealed-as-current = 0`, `not-yet-effective-as-current = 0`, `overruled-as-good-law = 0`. Online: weekly human review of sampled live answers, abstention-drift alarms, validation-failure dashboards, language/jurisdiction parity thresholds, latency/cost SLOs.

**Stress-test (three surfaces):** #1 ingestion (corpus poisoning, fake amending instruments, adversarial Gazette text, OCR attacks); #2 query (jailbreaks / injection in Devanagari/Romanized/mixed, PII extraction, cost amplification); #3 retrieval (injection embedded in statutory text aimed at the generator; canary-token detection on context). Gate = **no regression on the append-only suite AND a fresh red-team round per release**. Suite is append-only, owned by someone other than the releasing engineer, seeded continuously from live attack traffic. Coverage reported by threat-taxonomy cells (OWASP-LLM × input language form × attack surface), not payload count.

---

## 11. Observability & privacy

Full OTel trace (retrieval, temporal, model, prompt, index & source versions) — but legal query text is intent-revealing and sensitive. Rules: traces store **hashes/IDs by default**; raw query text and retrieved spans live only in a separate, access-controlled, short-retention store (≈30 days) linked by `trace_id`. **Redaction at the collector**, not the destination. Raw-content access = named roles, audited, dual-control (same rigor as admin lifecycle writes). Assume everything stored is breachable/subpoena-able; minimization is the control, independent of the pending compliance review's conclusions.

---

## 12. Tech stack (swappable) → role mapping

The **role** is fixed by §2; the **product** below is a default and may change after a bake-off.

| Role (fixed) | Default (swappable) | Hard constraint |
|---|---|---|
| Authority store | PostgreSQL | bitemporal (range types + exclusion constraints), the only source of truth |
| Search derivative | OpenSearch (BM25 + kNN) | derivative only; revalidated against authority |
| Embeddings | multilingual model strong on Devanagari | chosen by eval bake-off, versioned, re-embed only affected components |
| LLM gateway | any capable model | read-only tool access; never writes citations; never mutates |
| Orchestrator | any agent framework | state outside framework memory; allowlisted read-only tools; caps enforced |
| Tracing | OpenTelemetry | redaction at collector; minimal retention |
| Calendar | canonical BS↔AD table | versioned data with provenance, not a library |

---

## 13. Build phases

- **Phase 0 — Foundation.** Corpus ingested by hand-inspection; bitemporal skeleton; dumb baseline (retrieval-only, no reranker) end-to-end; eval + stress harness live. *Get one statute + one as-of query + one citation perfect before widening.*
- **Phase A — Statute path.** Both gates, single→per-claim as-of, deterministic RAG, zero-tolerance temporal gates green.
- **Phase B — Orchestrator + resilience.** Multi-hop with per-claim as-of, full degraded-mode ladder, chaos drills.
- **Phase C — Language hardening.** Romanized slice with its own target, translation-divergence check, parity thresholds.
- **Phase D — Precedent.** Holding-level precedent model + `overruled-as-good-law = 0`. **Case law answers ship here, not before.**

Eval and stress run continuously from Phase 0; nothing ships past its zero-tolerance gate.

---

## 14. Production Safety Annex (numbered, enforced requirements)

These are requirements, not aspirations. Each maps to a test.

- **PS-1** Precedent status is derived at holding level, from competent-bench relations, as-of the answer date. Never a stored case-level boolean. Gate: `overruled-as-good-law = 0`.
- **PS-2** `राजपत्रमा सूचना` commencement forces `not_yet_effective` + pending-notification display state + a commencement_dependency link. No fabricated effective date.
- **PS-3** Citations resolve to the authoritative instrument chain; consolidations are labeled `derived`. CI reconstructs every expression from base + effects over the normalized form or it does not publish.
- **PS-4** Store norm level, competence list, enabling-power link, and court `declared_invalid` events. Do not model vires as truth; disclose `in_force` ≠ validity.
- **PS-5** BS↔AD is versioned canonical data; boundary windows warn; future-range dates abstain.
- **PS-6** As-of is per-claim; comparative queries carry a set; each claim validates against its declared as-of.
- **PS-7** Abstention is server-owned; model self-abstention is advisory only.
- **PS-8** Romanized Nepali is a first-class eval slice with its own numeric target.
- **PS-9** Translation divergence is checked (back-translation/NLI), routed to reviewers, surfaced to users; never auto-corrected.
- **PS-10** OCR/source-kind provenance renders as a citation reliability badge.
- **PS-11** Postgres-down = no validated answers; revalidation replica is synchronous or freshness-gated against the lifecycle-write watermark.
- **PS-12** Stress gate = append-only suite no-regression AND fresh per-release red-team; coverage by taxonomy cells.
- **PS-13** LLM-judge κ ≥ 0.6, per-slice and rolling; below-threshold slices get 100% human review.
- **PS-14** Traces store IDs/hashes by default; raw content in a separate short-retention, dual-control store; redaction at collector.
- **PS-15** Status enum distinguishes repealed / spent / lapsed.
- **PS-16** Provisos and स्पष्टीकरण co-retrieve with their operative clause (eval-asserted).
- **PS-17** Retroactive amendments: close prior valid_time at the retroactive date (valid_time < transaction_time by design), flag affected historical answers, and treat user-notification of retroactively-changed prior answers as a product duty. Regression test straddles both axes.
- **PS-18** Edge cases under test: stillborn statute (repealed before commencement notification), code-switched romanized+English query bucket decided explicitly.