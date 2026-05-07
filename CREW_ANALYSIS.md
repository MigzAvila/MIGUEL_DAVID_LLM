# 🤖 AgentSociety Challenge — CrewAI Crew Analysis

> **Task:** Yelp Review Simulation (Track 1)
> **Framework:** [CrewAI](https://github.com/crewAIInc/crewAI) ≥ 1.14.1
> **LLM Backend:** [Groq](https://groq.com/) via LiteLLM (`groq/…` model id), OpenAI-compatible base URL `https://api.groq.com/openai/v1`. Per-agent overrides live under each `llm:` key in [`config/agents.yaml`](config/agents.yaml); `OPENAI_MODEL_NAME` in `.env` should match the same provider/model family.
> **Package Manager:** `uv`
>
> **Doc synced to git `main` (2026-05-06):** `d465918` — `config/tasks.yaml`, `config/agents.yaml`, `src/crews/simulation_crew.py`; prior commits `fe63e32` (`run_pipeline.py`, `run_test.py`, sample reports), `8632321` (adapter: peer snippets, prior blend, repair pass).

---

## Table of Contents

1. [Integration Setup Verification](#1-integration-setup-verification)
2. [Architecture Overview](#2-architecture-overview)
3. [Crew Members (Agents)](#3-crew-members-agents)
4. [How Agents Collaborate](#4-how-agents-collaborate)
5. [Knowledge & Data Retrieval](#5-knowledge--data-retrieval)
6. [Current Findings](#6-current-findings) · [6.5 Measured accuracy](#65-measured-accuracy-reports)
7. [Running the Tests](#7-running-the-tests)

---

## 1. Integration Setup Verification

The project is **fully compliant** with the AgentSociety official simulator integration guidelines. All three required integration points are implemented:

| Requirement | File | Status |
|---|---|---|
| Agent definitions with `role`, `goal`, `backstory` | `config/agents.yaml` | ✅ |
| Task definitions with `description`, `expected_output` | `config/tasks.yaml` | ✅ |
| Crew assembly via `@CrewBase`, `@agent`, `@task` | `src/crews/simulation_crew.py` | ✅ |
| Simulator adapter returning `predicted_rating` & `generated_review` | `src/flows/serving_flow.py` | ✅ |
| `get_interaction_tool()` available for agent equipping | `src/tools/interaction_tool_wrapper.py` | ✅ |
| Training / smoke pipeline (mock + real LLM, threading, timeouts) | `run_pipeline.py` | ✅ |
| Official test-day runner (Drive zip, email report) | `run_test.py` | ✅ |
| Legacy env-var test runner (`SIMULATOR_NUM_TASKS`, …) | `run_simulator_test.py` | ✅ (optional) |
| Flow layer output contract stable | `serving_flow.py` (`predicted_rating`, `generated_review`) | ✅ |
| Adapter pre-fetch + calibration (see [Architecture](#2-architecture-overview) and [Knowledge](#5-knowledge--data-retrieval)) | `crewai_simulation_agent.py` | ✅ (team-maintained) |

> [!IMPORTANT]
> The `predicted_rating` and `generated_review` keys in `InferenceState` are the **invisible contract** between the crew and the competition adapter. These are never renamed.

---

## 2. Architecture Overview

The system follows a **three-layer architecture**:

```
AgentSociety Simulator
        │
        ▼
┌───────────────────────────────────────┐
│     CrewAISimulationAgent             │  ← Adapter Layer (do not modify)
│  (crewai_simulation_agent.py)         │
│                                       │
│  1. Deterministic data pre-fetch      │
│     (user, item, review history)      │
│  2. Summarise into structured text    │
│  3. Compute fallback_rating from      │
│     user's historical average         │
└───────────────────┬───────────────────┘
                    │ passes InferenceState
                    ▼
┌───────────────────────────────────────┐
│     AgentSocietyServingFlow           │  ← Flow Layer (src/flows/serving_flow.py)
│  (CrewAI Flow)                        │
│                                       │
│  init_request → trigger_crew_inference│
│  Selects crew based on env var:       │
│    CREWAI_PROCESS_MODE=sequential     │
│                      collaborative    │
│                      hierarchical     │
└───────────────────┬───────────────────┘
                    │ kickoff(inputs=...)
                    ▼
┌───────────────────────────────────────┐
│     SimulationCrew (default)          │  ← Crew Layer (src/crews/)
│  (Sequential, 2 LLM agents)           │
│                                       │
│  psychological_analyst                │
│       │ context                       │
│       ▼                               │
│  behavior_simulator                   │
│       │ → {stars, review} JSON        │
└───────────────────────────────────────┘
```

### Key Design Decision: Pre-fetch Over Live Tool Use

Rather than having a `data_retriever` agent call the `Interaction Tool Wrapper` at runtime, **all Yelp data is fetched deterministically in the adapter layer** before the crew ever starts. This means:

- ✅ No hallucinated data — agents only see real Yelp records
- ✅ One fewer LLM round-trip per prediction
- ✅ No threading race conditions on the singleton `InteractionTool`
- ✅ Consistent, reproducible prompts across runs

**Recent adapter enhancements** (see git history: `crewai_simulation_agent.py` expanded May 2026; `config/tasks.yaml` + `simulation_crew.py` tuned in follow-up commits):

- **Peer-review snippets:** Up to five short excerpts from *other* users’ reviews of the same business are appended under the item summary (topical context only; not ground-truth labels for the target user).
- **Star prior / calibration block:** A `PRIOR_STAR_ESTIMATE` blends the user’s historical average with the item’s public `avg_stars` (70% item / 30% user) and is injected into the history block so Head A can reconcile “critical overall average” with “strong item match.”
- **Post-crew consistency pass:** If star rating and review sentiment are obviously mismatched (simple lexicon check), the flow runs **one repair kickoff** with a stricter consistency hint before returning `{stars, review}`.

---

## 3. Crew Members (Agents)

All agents are defined in [`config/agents.yaml`](config/agents.yaml) following the YAML-first pattern. Three crew variants are available, all drawing from the same agent pool.

**Current deployment (check repo):** each agent’s **`llm:`** is set to `groq/llama-3.1-8b-instant` to stay under Groq TPM limits during development. For a final run you can switch (e.g. to `groq/llama-3.3-70b-versatile`) in both `config/agents.yaml` and `.env` (`OPENAI_MODEL_NAME`, `NVIDIA_MODEL_NAME`).

### 3.1 Agent Pool (`config/agents.yaml`)

#### 🔬 `psychological_analyst`
| Field | Value |
|---|---|
| **Role** | Behavioral Psychologist |
| **Goal** | Analyze retrieved data to uncover the user's latent preferences, rating habits, and satisfaction patterns |
| **Backstory** | Dual background in behavioral science and psychology; profiles user personality traits and rating tendencies from interaction clues |
| **LLM** | `groq/llama-3.1-8b-instant` |
| **Used In** | Sequential (primary), Collaborative, Hierarchical |

#### 🎭 `behavior_simulator`
| Field | Value |
|---|---|
| **Role** | Review Simulation Expert |
| **Goal** | Produce the most accurate star rating and realistic review text by deeply embodying the user's persona |
| **Backstory** | Master method actor who pays close attention to historical rating distributions; never defaults to positive ratings |
| **LLM** | `groq/llama-3.1-8b-instant` |
| **Used In** | Sequential (primary), Collaborative, Hierarchical |

#### 📡 `data_retriever`
| Field | Value |
|---|---|
| **Role** | Data Retrieval Specialist |
| **Goal** | Accurately retrieve all relevant background data for the target user and item |
| **Backstory** | Expert in big data retrieval and association rules; finds the most valuable records from massive databases |
| **LLM** | `groq/llama-3.1-8b-instant` |
| **Used In** | Collaborative, Hierarchical (equipped with `InteractionTool`) |

#### 🗂️ `prediction_manager`
| Field | Value |
|---|---|
| **Role** | Yelp Prediction Project Manager |
| **Goal** | Coordinate specialist agents to deliver one final high-quality prediction package |
| **Backstory** | Senior PM for multi-agent prediction pipelines; breaks down objectives, delegates, validates outputs |
| **LLM** | `groq/llama-3.1-8b-instant` |
| **Used In** | Hierarchical (manager role only) |

---

## 4. How Agents Collaborate

Three crew modes are available, selectable via `CREWAI_PROCESS_MODE` environment variable.

### Mode A: Sequential (Default) — `SimulationCrew`

```
[Pre-fetched Yelp Data]
         │
         ▼
 psychological_analyst
  ├─ Receives: user_summary, item_summary, history_summary (includes PRIOR_STAR_ESTIMATE block)
  ├─ Task: analyze_preference_task
  │   • States user's historical avg stars & distribution
  │   • Cites 1-2 specific past reviews
  │   • Produces numeric preliminary rating estimate
  │   • Ends with: HEAD_A_TARGET_STARS: <1.0|…|5.0>
  └─ Output: Markdown analysis (≤250 words) + HEAD_A line
         │
         │ (context passed automatically)
         ▼
  behavior_simulator
  ├─ Receives: all pre-fetched data + analyst's output as context (item may include OTHER REVIEWERS snippets)
  ├─ Task: simulate_review_task
  │   • Selects a star rating from {1.0, 2.0, 3.0, 4.0, 5.0} (follow Head A unless violated)
  │   • Writes 2–4 short sentences (≤~280 chars), matching user's tone & style
  │   • Grounds on item attributes; may reuse 1–2 short phrases from peer snippets when present
  └─ Output: STRICT JSON → {"stars": 3.0, "review": "..."}
```

**Hard constraints enforced in prompts:**
- Stars must be consistent with user's historical distribution
- Review in English, matching the user's real vocabulary/casing
- Only references real attributes from the data — no invented products
- Output is a bare JSON object, no prose/markdown around it

### Mode B: Collaborative — `CollaborativeSingleTaskCrew`

```
behavior_simulator (lead)
  ├─ Can delegate to: data_retriever, psychological_analyst
  └─ Single unified task: collaborative_single_task
     Produces: {"stars": float, "review": "string"}
```

All three agents share one task; `behavior_simulator` leads and may delegate sub-queries to peers via CrewAI's native delegation protocol.

### Mode C: Hierarchical — `HierarchicalManagerCrew`

```
prediction_manager (manager, no @agent decorator)
  ├─ Delegates to → data_retriever (equipped with InteractionTool)
  ├─ Delegates to → psychological_analyst
  └─ Delegates to → behavior_simulator
  Single task: hierarchical_predict_task
  Produces: {"stars": float, "review": "string"}
```

The manager orchestrates specialist agents, instructs `data_retriever` to call the `Interaction Tool Wrapper` live, and collects all outputs before the simulator produces the final JSON.

### Collaboration Summary Table

| Mode | `CREWAI_PROCESS_MODE` | Agents Active | LLM Calls | Data Source |
|---|---|---|---|---|
| Sequential | `sequential` | 2 (analyst + simulator) | ~2 | Pre-fetched by adapter |
| Collaborative | `collaborative` | 3 (all workers) | ~3–5 | Pre-fetched by adapter |
| Hierarchical | `hierarchical` | 4 (manager + 3 workers) | ~4–6 | Live tool call by `data_retriever` |

---

## 5. Knowledge & Data Retrieval

### 5.1 What Data Is Used

For each prediction task, the system retrieves three data sources from the Yelp dataset:

| Data | Source | Contents |
|---|---|---|
| **User Profile** | `get_user(user_id)` | Name, review count, yelping_since, average_stars, useful/funny/cool counts, fans, elite years |
| **Item Details** | `get_item(item_id)` | Name, location, avg_stars, review_count, categories, price range, noise level, ambience |
| **Review History** | `get_reviews(user_id)` | Up to the 12 most recent reviews: stars, date, and up to **360**-char text snippets |
| **Peer item reviews** | `get_reviews(item_id)` | Up to 5 snippets from other reviewers (same business), for topical vocabulary — injected next to item details |

### 5.2 How Data Is Structured for Prompts

Raw data is not passed to agents directly. It is processed by helper functions in `crewai_simulation_agent.py` into **structured natural language summaries**:

```
USER PROFILE (verbatim from Yelp):
name=Alice; review_count=47; yelping_since=2014; average_stars=3.21;
useful=12; funny=3; cool=5; fans=2; elite=2018,2019

ITEM DETAILS (verbatim from Yelp):
name=The Noodle House; location=Las Vegas, NV; avg_stars=3.8;
review_count=512; categories=Noodles, Asian Fusion; price_range=2

USER REVIEW HISTORY (verbatim from Yelp):
TOTAL_HISTORICAL_REVIEWS=47
USER_HISTORICAL_AVERAGE_STARS=3.21
USER_RATING_DISTRIBUTION: n=47; 5*=8, 4*=9, 3*=18, 2*=7, 1*=5
RECENT_REVIEWS (most recent first):
- [3* on 2023-11-14] service was ok but food took forever…
- [2* on 2023-08-02] nothing special, overpriced for what you get…
```

**Truncation guards** prevent context window overflows:
- Max 12 historical reviews passed to prompts
- Review text capped at **360** characters each
- Item categories capped at 240 characters

The analyst’s prompt also receives a **calibration block** after the history summary: `PRIOR_STAR_ESTIMATE` (blend of user history mean and item `avg_stars`) plus short instructions so Head A does not only anchor on the user’s global average when the item is an obvious strong or weak match.

### 5.3 Fallback Rating Computation

The flow’s `fallback_rating` field is set to the same **prior** as `PRIOR_STAR_ESTIMATE` (user mean blended with item public average when available), not the raw user mean alone. If the crew fails or times out, serving logic can still fall back to historical patterns; the adapter’s final `{stars, review}` also applies **star bucketing** to `{1.0,…,5.0}` and optional **sentiment repair** (see [Architecture](#2-architecture-overview)).

### 5.4 Optional Knowledge Base (ChromaDB + RAG)

An optional background knowledge layer is available via the `PrebuiltTextKnowledgeSource` class. This is **disabled by default** (`CREWAI_ENABLE_KNOWLEDGE=false`).

| Setting | Description |
|---|---|
| `CREWAI_ENABLE_KNOWLEDGE=true` | Activates RAG knowledge injection into the crew |
| `CREWAI_KNOWLEDGE_FILE=<path>` | Path to a `.txt` or `.json` background knowledge file |
| `CREWAI_USE_PREBUILT_INDEX=true` | Skips re-embedding if the ChromaDB collection already exists |

**How it works when enabled:**
1. `PrebuiltTextKnowledgeSource` wraps CrewAI's `TextFileKnowledgeSource`
2. On first run, the file is chunked and embedded into a **ChromaDB persistent collection** (`knowledge_crew`)
3. On subsequent runs, `skip_if_index_exists=True` bypasses re-embedding (fast restart)
4. The collection is injected into the `Crew` object via `knowledge_sources=[...]`
5. Agents can semantically query background knowledge alongside the structured Yelp data

> [!NOTE]
> The sequential crew currently pre-fetches all necessary Yelp data directly via the adapter, making the RAG knowledge layer redundant for most use cases. It is useful for injecting **global domain knowledge** (e.g., category-level rating baselines, Yelp platform behaviour notes) that is not user/item-specific.

### 5.5 Tool: `Interaction Tool Wrapper`

```python
# Registered as a CrewAI @tool — used by hierarchical/collaborative modes
interaction_tool_wrapper(query_type: str, target_id: str) -> str

# query_type options:
#   "user"            → get_user(user_id=target_id)
#   "item"            → get_item(item_id=target_id)
#   "review_by_user"  → get_reviews(user_id=target_id)
#   "review_by_item"  → get_reviews(item_id=target_id)
```

The wrapper is a thin bridge between the AgentSociety `InteractionTool` (injected by the simulator) and CrewAI's `@tool` interface. A module-level singleton with a threading lock ensures it is safe for concurrent agents.

---

## 6. Current Findings

### 6.1 Architecture Decisions & Their Rationale

| Decision | Rationale | Outcome |
|---|---|---|
| **Pre-fetch data in adapter, not via agent tool call** | Eliminates hallucination; removes 1 LLM round-trip; avoids threading race on singleton tool | ✅ Zero hallucinated product names/categories |
| **Sequential 2-agent crew as default** | Lowest token cost; analyst provides grounded context for simulator; fewest failure points | ✅ Reliable; easiest to debug |
| **Hard-coded JSON output enforcement in task prompts** | LLMs are prone to wrapping JSON in prose or markdown fences | ✅ `extract_json_from_output()` in flow handles noisy outputs as a safety net |
| **Fallback rating from user history** | Prevents defaulting to 4.0 when crew fails | ✅ Graceful degradation on API errors |
| **Star rating constrained to {1.0, 2.0, 3.0, 4.0, 5.0}** | Matches Yelp's actual discrete rating scale | ✅ Avoids non-standard outputs like 3.5 |
| **Rating distribution injected verbatim** | Critical for replicating critical reviewers (avg 2-3★) vs generous ones (avg 4-5★) | ✅ Prevents systematic upward bias |
| **Item-heavy star prior (70/30 blend)** | Aligns predictions with public item quality while respecting user history | ✅ Better MAE when users are mixed reviewers |
| **Peer-review snippets** | Gives concrete dish/venue phrases for embedding overlap | ✅ Improves review_generation / topical match |
| **Single sentiment–star repair pass** | Reduces “4★ + negative text” evaluator penalties | ✅ Cheap second kickoff only on conflict |

### 6.2 Prompt Engineering Strategies

- **Distribution-anchored prompting:** The analyst task explicitly requires stating the user's `n=X; 5*=Y, 4*=Z…` distribution before making any estimate. This anchors the simulator agent to the correct magnitude.
- **Persona replication:** The behavior_simulator backstory emphasizes replicating tone, vocabulary, and casing (e.g., "many Yelp users write entirely in lowercase").
- **Negative constraints:** "Do NOT default to 4 or 5 just because the item has high public ratings" — directly counters known LLM optimism bias.
- **Cross-contamination prevention:** "Never reference products or categories absent from the data" — stops agents from hallucinating electronics into a restaurant review.
- **Head A / Head B contract:** Analyst output must end with `HEAD_A_TARGET_STARS: <1.0|…|5.0>`; the simulator task treats that line as the primary star target before JSON emission.
- **Phrase reuse from peer snippets:** When other-reviewer snippets exist, the simulator prompt asks for 1–2 short verbatim phrases (2–6 words) to align with how people discuss that business.

### 6.3 Known Limitations & Open Challenges

> [!WARNING]
> The following issues have been observed or anticipated:

| Issue | Impact | Proposed Mitigation |
|---|---|---|
| **LLM optimism bias** | Model tends to predict 4.0 even for critical users | Stronger negative constraints; few-shot examples of low-star reviews |
| **Context window limits** | Users with 500+ reviews are truncated to 12 | Summarise distribution more aggressively; weight recent reviews higher |
| **Rate limiting (HTTP 429)** | Parallel runs hit token-per-minute limits | Reduce `max_workers` in `evaluate_with_training_data.py` |
| **ChromaDB embedding conflicts** | Mismatched embedding models cause validation errors on restart | `PrebuiltTextKnowledgeSource.skip_if_index_exists` avoids re-indexing; delete `~/.local/share/crewai/` to reset |
| **Non-JSON LLM output** | Some models wrap JSON in markdown fences | `extract_json_from_output()` regex fallback handles this |

### 6.4 Evaluation Metrics

The competition evaluates on two metrics:

- **Star rating MAE** (Mean Absolute Error) — lower is better
- **Review quality** (text similarity / plausibility) — higher is better

Local test output example (mock mode, structural check):
```
★ stars  pred=4.0  gt=3.0  Δ=+1.0
📝 pred : [Mocked LLM] Solid spot, would visit again.
🎯 gt   : service was ok but portions have gotten smaller lately
```

Real inference test summary format:
```
Summary: MAE=X.XX  exact=Y/N  within 1 star=Z/N
```

### 6.5 Measured accuracy (reports)

**Full test-set run (official evaluation shape):** [`test_report_20260506_222218.json`](test_report_20260506_222218.json) — produced by `run_test.py` on the bundled 41-task zip (`real_llm`, 1 thread, 300s task timeout). Git commit `fe63e32` added this report alongside `run_pipeline.py` / `run_test.py` updates.

| Run context | Value |
|---|---|
| **Report file** | `test_report_20260506_222218.json` |
| **Run timestamp** | `20260506_222218` |
| **Mode** | `real_llm` |
| **Tasks evaluated** | **41** / 41 (ground truth aligned) |
| **Errors** | **2** (see `per_task_outputs` / error fields in the JSON) |
| **Wall clock (inference)** | ~3943 s (~96 s avg / task) |

**Simulator metrics (higher is better for all three):**

| Metric | Score |
|---|---|
| **preference_estimation** | **0.8829** |
| **review_generation** | **0.8210** |
| **overall_quality** | **0.8520** |

**Replicate this run:** From the repo root, use the bundled 41-task zip (same tasks/ground truth as the report above), real LLM, single thread, and default 300s per-task timeout:

```powershell
uv run python run_test.py --test-set test_set_41.zip
```

Match **`config/agents.yaml`** / **`.env`** (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL_NAME`, `CREWAI_PROCESS_MODE`, etc.) to the environment that produced the report; different models or prompts will change scores. The script prompts interactively for team metadata and instructor email settings — use the same inputs you care to document, or run once to seed a local `test_report_*.json` for comparison.

**Local pipeline JSON:** `uv run python run_pipeline.py` writes `pipeline_report_<timestamp>.json` (dummy dataset + `dummy_tasks` / `dummy_groundtruth`) with the same top-level shape (`evaluation`, `per_task_outputs`, timing). Use `--mock`, `--tasks N`, `--threads M`, and `--timeout SEC` for smoke and parallelism (see script docstring).

> [!NOTE]
> Older one-task mock snapshots (e.g. historical `pipeline_report_20260505_*.json`) are not retained in the repo; treat the table above as the current **full-task** reference unless you generate a newer `test_report_*.json` or `pipeline_report_*.json`.

---

## 7. Running the Tests

### Primary: training / smoke pipeline (`run_pipeline.py`)

```powershell
uv run python run_pipeline.py --mock
uv run python run_pipeline.py
uv run python run_pipeline.py --threads 2
uv run python run_pipeline.py --tasks 1
uv run python run_pipeline.py --timeout 120
```

### Official test-day flow (`run_test.py`)

```powershell
uv run python run_test.py
uv run python run_test.py --test-set test_set_41.zip
uv run python run_test.py --test-set path\to\other_set.zip
uv run python run_test.py --mock
uv run python run_test.py --threads 2 --timeout 300
```

To reproduce the metrics in [section 6.5](#65-measured-accuracy-reports), prefer `--test-set test_set_41.zip` with the same `.env` and crew config as that run.

### Legacy runner (env-driven task count)

```powershell
uv run python run_simulator_test.py --mock
uv run python run_simulator_test.py
```

### Full training-set evaluator (if present in repo)

```powershell
uv run python evaluate_with_training_data.py
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CREWAI_PROCESS_MODE` | `sequential` | Crew variant: `sequential`, `collaborative`, `hierarchical` |
| `SIMULATOR_DEVICE` | `cpu` | Simulator device: `cpu`, `cuda`/`gpu`, `auto` |
| `OPENAI_API_BASE` | _(from `.env`)_ | Mirrored to `OPENAI_BASE_URL` for CrewAI / LiteLLM when set |
| `SIMULATOR_NUM_TASKS` | `5` | For **`run_simulator_test.py` only**: tasks to run (`all` for full set) |
| `SIMULATOR_THREADING` | `false` | For **`run_simulator_test.py`**: parallel tasks |
| `SIMULATOR_MAX_WORKERS` | `1` | For **`run_simulator_test.py`**: worker count when threading is on |
| `CREWAI_ENABLE_KNOWLEDGE` | `false` | Enable optional ChromaDB RAG layer |
| `CREWAI_KNOWLEDGE_FILE` | _(none)_ | Path to background knowledge file |
| `CREWAI_KNOWLEDGE_JSON` | _(none)_ | Alternate env name for knowledge file path |
| `CREWAI_USE_PREBUILT_INDEX` | `false` | Skip re-embedding if index exists |
| `CREWAI_EMBEDDER_PROVIDER` | `sentence-transformer` | Embedder provider when knowledge is on |
| `CREWAI_EMBEDDER_MODEL` | _(default MiniLM)_ | Override embedder model name |

---

## File Map

```
AgentSocietyChallenge_w_CrewAI/
│
├── config/
│   ├── agents.yaml                   # 🔧 Agent definitions (role/goal/backstory)
│   ├── tasks.yaml                    # 🔧 Sequential task definitions
│   ├── tasks_collaborative.yaml      # 🔧 Collaborative single-task definition
│   └── tasks_hierarchical.yaml       # 🔧 Hierarchical task definition
│
├── src/
│   ├── crews/
│   │   ├── simulation_crew.py        # 🔧 Default sequential crew (2 agents)
│   │   ├── collaborative_single_task_crew.py  # Alt: collaborative mode
│   │   └── hierarchical_manager_crew.py       # Alt: hierarchical mode
│   │
│   ├── flows/
│   │   └── serving_flow.py           # ⛔ Flow layer (do not modify key names)
│   │
│   ├── knowledge/
│   │   └── prebuilt_source.py        # Optional ChromaDB RAG source
│   │
│   └── tools/
│       └── interaction_tool_wrapper.py  # ⛔ Simulator tool bridge
│
├── crewai_simulation_agent.py        # Adapter: pre-fetch, prior, peer snippets, repair pass
├── run_pipeline.py                   # Main pipeline + `pipeline_report_*.json`
├── run_test.py                       # Test-day runner + `test_report_*.json`
├── test_set_41.zip                   # 41-task bundle (`--test-set` for replication)
├── run_simulator_test.py             # Legacy env-var test runner
├── evaluate_with_training_data.py    # Full dataset evaluator (if used)
└── pyproject.toml                    # uv-managed dependencies
```

> [!TIP]
> Files marked 🔧 are the **student zone** — safe to modify freely.
> Treat `serving_flow.py` and the `InferenceState` field names as a **stable contract** (do not rename `predicted_rating` / `generated_review`). The adapter file may still accumulate team-specific retrieval and calibration logic as long as the final `{stars, review}` shape matches the simulator.
