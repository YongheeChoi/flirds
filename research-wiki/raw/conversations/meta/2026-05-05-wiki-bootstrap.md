---
type: conversation
date: 2026-05-05
topic: meta
participants: [Yonghee, Claude]
tags: [wiki-setup, scope, conventions]
---

# 2026-05-05 — wiki bootstrap and scope expansion

The conversation that created this wiki. Distilled record of Yonghee's actual asks, decisions made, and conventions established.

## Session 1 — initial setup (auto mode)

**Yonghee shared** the "LLM Wiki" idea document (the Tobi Lütke pattern: raw sources + LLM-maintained wiki + schema file) and let Claude proceed in auto mode.

**State found**: `Research_wiki/` already had `raw/papers/flirds/` populated with 20 papers (mix of PDF and Obsidian-clipped markdown) and `raw/conversations/flirds/` empty. Topic: data valuation / attribution / influence.

**Decisions made**:
- Adopted layout: `wiki/{index,log,overview}.md` + `wiki/{sources,concepts,threads}/`.
- `CLAUDE.md` at root holds the schema and workflows.
- Two papers ingested as demo: [[sources/data-banzhaf]], [[sources/in-run-data-shapley]]. Pulled in 5 concept pages + 2 thread pages from those.
- Memory saved: user role, project structure, raw-papers reference.

## Session 1 mid — language preference

**Yonghee**: "지식은 영어로 저장해도 되는데 나와의 대화는 항상 한글로 해줘."

→ Saved as feedback memory: chat Korean, wiki/files English. Proper nouns and math notation kept in original form.

## Session 1 late — scope expansion and conversation persistence

**Yonghee's three asks, verbatim-ish**:

1. "raw 파일들 중에서 아직 ingest 안된게 있으면 ingest 해줘."
2. "나는 AI를 연구하는 대학원생이고 LLM과의 원활한 협업을 통해 연구를 진행하고 싶어. 그래서 너와의 대화가 raw 데이터에 지속적으로 업데이트 되었으면 해."
3. "지금 당장에 data valuation 연구를 하고 있지만 앞으로 더 많은 영역의 연구를 할 예정이야. 그러니 목적을 꼭 data valuation 연구에 두기 보단 AI field 연구로 넓혀놓는게 좋을 것 같아."
4. "나에 대해 정리해놓으면 좋을만한 사실들이 있을까?"

**Decisions and structural changes**:

- **Scope broadened**: CLAUDE.md, overview, and memories now frame the wiki as for "all of Yonghee's AI research over time," with `flirds` (data valuation) marked as the *current active topic* — not the wiki's permanent scope.
- **New directory layer**: `wiki/topics/<topic>.md` landing pages introduced. Existing data-valuation synthesis moved from `wiki/overview.md` into `wiki/topics/flirds.md`. `wiki/overview.md` is now a top-level hub listing active topics.
- **Conversation logging workflow added** to CLAUDE.md: distilled transcripts go to `raw/conversations/<topic>/YYYY-MM-DD-<slug>.md` (or `meta/` for wiki-meta). Insights from conversations get distilled into wiki pages; raw transcripts are preservation, wiki is curation.
- **Memory added**: `feedback_conversation_logging.md` — the rule above, with rationale ("Yonghee treats the wiki as a context-persistence mechanism").

**Open questions Yonghee was asked** (TBD):
1. PhD year, lab/advisor (or just institution).
2. Other AI research areas planned next.
3. Tools / frameworks (PyTorch, JAX, cluster setup).
4. Collaboration style preference (strong opinions vs. options; tolerance for pushback).

Whatever Yonghee answers should be saved as user-memory updates and ingested back into this conversation log.

## Open / pending at end of session

- 18 of 20 raw papers still un-ingested. Plan: 6 markdown clippings first, then 12 PDFs (need identification first).
- Etymology of `flirds` not confirmed — current best guess in [[topics/flirds]] is a placeholder.
- Need Yonghee's answers to the four introspective questions to enrich the user-memory.
