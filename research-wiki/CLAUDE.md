# Research_wiki — schema and workflow

This is a **personal research wiki** maintained collaboratively by Yonghee and an LLM agent. The pattern follows Tobi Lütke's "LLM Wiki" idea: raw sources are immutable; the wiki is an LLM-curated synthesis that grows with every source ingested and every question asked.

Yonghee reads the wiki in Obsidian; you write and maintain it.

## Purpose and scope

This wiki is the durable knowledge substrate for Yonghee's **Flirds project** — *Federated Learning + In-Run Data Shapley*, a federated data-valuation method extending IRDS to the FL setting with LoRA-based PEFT.

The wiki's domain is **data valuation, data attribution, and data influence in machine learning** — and specifically the slice of that field needed to inform the Flirds project. Coverage:

- **Semivalues**: Shapley, Banzhaf, LOO, Beta, CS, Asymmetric, DU.
- **Influence functions and gradient-based methods**: classical IF, TracIn, Datamodels, DataInf, LoGra/Logix, EKFAC.
- **Federated and decentralized data valuation**: GTG-Shapley, ShapleyFL, S-FedAvg, SPACE, Ripple Shapley, FedDQC, DICE, Federated Banzhaf, FedSV, ComFedSV (in scope as it appears).
- **LLM-scale data attribution**, **data markets / incentive design** — covered insofar as they inform Flirds' framing.

Other areas of Yonghee's research (e.g. interpretability, alignment, RL, multimodal — TBD) belong in **separate wikis**, not this one. Yonghee's stated reason: "research areas should not be mixed in one wiki." Treat scope creep as friction; if a topic doesn't inform the data-valuation discussion, it's out of scope here.

The **primary record** of the Flirds project's design decisions is `raw/conversations/flirds/conversation{1..4}.md` (Yonghee's design conversations with another LLM). Read those before answering project-specific questions — the wiki's flirds page is the distillation but the raw conversations are the source of truth.

## Layout

```
Research_wiki/
├── CLAUDE.md                       ← this file
├── raw/                            ← immutable sources, never edit
│   ├── papers/flirds/              ← PDFs and Obsidian-clipped markdown
│   └── conversations/
│       ├── flirds/                 ← Flirds-project design conversations (primary record)
│       └── meta/                   ← wiki-meta conversations (setup, scope, conventions)
└── wiki/                           ← LLM-maintained
    ├── index.md                    ← content catalog (every page listed)
    ├── log.md                      ← chronological append-only log
    ├── overview.md                 ← field synthesis + Flirds project framing
    ├── flirds.md                   ← Flirds project state: locked design decisions, open questions, experiment plan
    ├── sources/                    ← one page per ingested source (papers, conversations)
    │   └── <slug>.md
    ├── concepts/                   ← concept and method pages
    │   └── <slug>.md
    └── threads/                    ← cross-cutting research threads / open questions
        └── <slug>.md
```

`raw/papers/flirds/` is named after the project; that organization predates this CLAUDE.md and is kept as-is. The wiki itself is *not* further subdivided by topic — this wiki is single-purpose.

Filenames use `kebab-case.md`. Slugs should be short and stable.

## Page conventions

Every wiki page has YAML frontmatter:

```yaml
---
type: source | concept | thread | overview | index | log
title: <human title>
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [<list of source slugs that contributed>]   # for non-source pages
tags: [<freeform>]
---
```

Use Obsidian-style internal links: `[[concepts/shapley-value]]`, `[[sources/data-banzhaf]]`. Do not use bare paths like `concepts/shapley-value.md` — the `[[…]]` form lets Obsidian resolve and graph them.

When citing a claim, link to the source page: `[[sources/<slug>]]`. The source page itself links to the raw file in `raw/`.

Quote sparingly. Paraphrase with your own words. When you do quote, keep it under 15 words and use quotation marks. Never reproduce long passages from sources verbatim.

### Source pages

`wiki/sources/<slug>.md` summarizes one paper or document. Suggested sections:

- **Citation** — title, authors, venue/year, link to raw file.
- **TL;DR** — 2–3 sentences.
- **Problem** — what gap the source addresses.
- **Method** — the approach in concrete terms; include math sketch only when load-bearing.
- **Key results** — the empirical or theoretical claims that matter for our wiki.
- **Connections** — links to related concept and thread pages, with one-line notes (e.g. "extends [[concepts/data-shapley]] to the in-run setting").
- **Notes / open questions** — the user's reactions, things to double-check, follow-up reads.

### Concept pages

`wiki/concepts/<slug>.md` describes a method, definition, or idea (e.g. `shapley-value`, `influence-function`, `semivalue`, `dice`, `datainf`). Suggested sections:

- **One-line definition.**
- **Formal definition** — math when needed.
- **Intuition.**
- **Variants and history** — how it evolved, what it generalizes/specializes.
- **Strengths and limitations.**
- **Where it appears in the literature** — list of `[[sources/...]]` links with a one-line each.
- **See also** — sibling concept pages.

### Thread pages

`wiki/threads/<slug>.md` is for cross-cutting questions that span multiple sources — e.g. `retraining-vs-in-run-attribution`, `replication-robustness`, `fl-contribution-fairness`, `attribution-at-llm-scale`. These are the most valuable pages because they capture *synthesis*. Cite at least 3 sources in any thread page; flag contradictions explicitly.

## Workflows

### Ingest a source

When the user drops a file in `raw/` and says "ingest" (or just names the file):

1. **Read** the source. For PDFs, prefer reading the markdown sibling if one exists.
2. **Discuss** the highlights with the user briefly (2–4 bullets) and ask whether to emphasize anything specific. In auto mode, proceed with reasonable defaults but still surface the highlights.
3. **Create** `wiki/sources/<slug>.md` using the source-page template.
4. **Update** related concept pages — add this source to their "where it appears" lists, and revise text if the source changes the picture (new variant, contradicting result, sharper bound, etc.).
5. **Update** related thread pages — same logic. Create new threads if the source opens a new line of inquiry the wiki doesn't yet cover.
6. **Update** `wiki/index.md` — list the new source page, list any new concept/thread pages.
7. **Update** `wiki/overview.md` if the synthesis shifts.
8. **Append** to `wiki/log.md`:
   ```
   ## [YYYY-MM-DD] ingest | <source title>
   - source: [[sources/<slug>]]
   - touched: [[concepts/...]], [[threads/...]]
   - note: <one line on what this source added>
   ```

A single ingest typically touches 5–15 wiki files. Don't rush the cross-references — they are the value.

### Answer a query

When the user asks a question:

1. Read `wiki/index.md` to find candidate pages, then drill in.
2. If the wiki has the answer, synthesize it with `[[…]]` citations.
3. If the answer is partial, say what's known from the wiki and what's missing.
4. **File good answers back.** A useful comparison, a synthesis, a discovered connection — write it up as a new thread page or extend an existing one. Note in `log.md` as a `query | …` entry.

### Log a conversation

Yonghee's intent: chat conversations should persist into the wiki so that context bridges across separate Claude sessions. Workflow:

1. **Identify a substantive checkpoint** — end of a session, or after a meaningful decision/insight. Trivial back-and-forth doesn't need logging.
2. **Write a distilled transcript** to `raw/conversations/<topic>/YYYY-MM-DD-<slug>.md` (or `raw/conversations/meta/...` for wiki-structure conversations not tied to research). Use this format:
   ```yaml
   ---
   type: conversation
   date: YYYY-MM-DD
   topic: <topic-slug or "meta">
   participants: [Yonghee, Claude]
   tags: [<freeform>]
   ---
   ```
   Body: a faithful but compressed record. Include Yonghee's actual asks, decisions made, and any of Yonghee's stated preferences or directions. You can paraphrase Claude's responses; preserve Yonghee's words more carefully.
3. **Distill into the wiki** — pull durable insights into the relevant existing pages: add a "from conversation [date]" attributed note to the right concept/thread, expand open questions, or create a new thread page if a substantial new line of inquiry emerged. The raw transcript is preservation; the wiki is curation.
4. **Append to log**: `## [YYYY-MM-DD] conv | <subject>` with `- raw: raw/conversations/...`, `- distilled into: [[…]]`, `- note: <one line>`.
5. **Update memory** if the conversation surfaced a stable preference, a long-running fact about Yonghee, or a recurring workflow change. Conversations are ephemeral; memory is for the patterns underneath.

Decisions about the wiki itself (scope, conventions, structure) should be logged under `meta` rather than under a research topic.

### Lint pass

When asked to lint or health-check:

- Find orphan pages (no inbound `[[…]]` links).
- Find stub pages (under ~80 words) and either expand or merge.
- Find concepts mentioned across ≥3 sources but lacking a dedicated page; create one.
- Find contradictions between pages and flag them on the relevant thread page.
- Suggest follow-up sources to look for or web searches to run.
- Append a `## [date] lint | …` entry to `log.md` summarizing what changed.

## Log conventions

Every log entry starts with `## [YYYY-MM-DD] <kind> | <subject>` so `grep "^## \[" wiki/log.md` works. Kinds: `ingest`, `query`, `lint`, `note`.

Newest entries at the **bottom** (append-only).

## Style

- Be terse. The user reads in Obsidian; verbosity hurts.
- Prefer crisp definitions and tight comparisons over rambling exposition.
- Math is welcome where it earns its keep. Use LaTeX inline (`$…$`) and block (`$$…$$`) — Obsidian renders both.
- When two sources disagree, present both and name the disagreement; do not paper over it.
- "I don't know" is a legitimate answer. Mark gaps explicitly with `> TODO: …` so a future ingest can fill them.

## Operating mode

The user is an active collaborator, not a bystander. Default to **one source at a time**, surfacing key takeaways before writing pages, so the user can steer emphasis. Auto mode relaxes this — proceed with sensible defaults and let the user course-correct.

Never edit `raw/`. Never delete pages without checking inbound links first.
