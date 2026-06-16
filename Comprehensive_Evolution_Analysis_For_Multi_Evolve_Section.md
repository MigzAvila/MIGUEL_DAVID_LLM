# Comprehensive Evolution Analysis: WWW'25 AgentSociety Challenge

## 1. Introduction & Methodology

Our objective was to construct and autonomously optimize a multi-agent system capable of flawlessly predicting user behavior and writing realistic reviews based on factual public data. To achieve this, we combined the **CrewAI** framework with the **OpenEvolve** Multi-Agent Evolutionary Algorithm (MAEA) library. 

### Why We Decoupled the Configuration (YAML-First Architecture)
Standard CrewAI implementations hardcode the `role`, `goal`, and `backstory` of agents directly into Python decorators (`@agent`, `@task`). This makes automated prompt evolution nearly impossible because the evaluator cannot safely mutate live Python code without breaking syntax. 
We completely decoupled the system by storing the prompt configurations in `agents_evolving.yaml` and `tasks_evolving.yaml`. This allowed OpenEvolve to hot-swap these files at runtime via the `OPENEVOLVE_AGENTS_YAML` environment variable, enabling rapid, syntax-safe mutation and evaluation across hundreds of LLM generations.

### The Evaluator & Scoring Mechanics
The `websocietysimulator` grades the AI on two primary metrics:
1. **Preference Estimation (`preference_estimation`)**: Measures how close the agent's predicted star rating is to the user's actual historical rating.
2. **Review Generation (`review_generation`)**: Evaluates the semantic similarity (via LLM topic embedding clusters), vocabulary overlap, and style match of the generated text compared to the ground truth.
3. **Combined Score (`overall_quality`)**: The ultimate metric optimized during the baseline evolution, requiring a delicate balance between rigid mathematical accuracy (stars) and organic, human-like creativity (text).

---

## 2. In-Depth Analysis: The Evolved AI Strategies

Over 50 iterations of evolutionary mutation, the `minimax-m2.7` model discovered advanced prompting strategies that a human engineer would likely miss. 

### Strategy 1: The "Standard Deviation" Classifier (Agents)
In the initial baseline, agents were instructed simply to "predict the stars." OpenEvolve mutated the `psychological_analyst` agent's prompt to enforce rigorous statistical bounds. 
The best evolved `agents_evolving.yaml` explicitly commands the LLM to:
> "Identify rating consistency: low std dev (< 0.8) = predictable rater, high std dev (> 1.2) = variable rater."
> "When std dev is low (< 0.8), the user is predictable—stick close to mean. When std dev is high (> 1.2), allow more flexibility but still center predictions on mean. Never default to 4 stars."

**Why this worked**: By forcing the LLM to calculate the standard deviation first, it prevented the model from defaulting to a "safe" 4.0 or 5.0 rating on highly rated businesses when the specific user is historically a harsh critic.

### Strategy 2: The Rating Calculation Formula (Tasks)
During the independent task evolution (which achieved 0.9702), OpenEvolve discovered that the LLM needed a mathematical formula to reconcile conflicting data. It mutated the `analyze_preference_task` to include an exact formula:
> **RATING CALCULATION FORMULA:**
> 1. Extract USER_AVG = user's historical average star rating
> 2. Extract ITEM_AVG = item's average star rating (if available)
> 3. Calculate PRIOR_STAR_ESTIMATE = (USER_AVG × 0.7) + (ITEM_AVG × 0.3) when ITEM_AVG exists
> 4. CRITICAL: Final HEAD_A_TARGET_STARS MUST be within ±1.0 stars of USER_AVG

**Why this worked**: The 70/30 blend mathematically forces the LLM to prioritize the user's historical bias (70%) over the business's public popularity (30%), perfectly aligning with the evaluator's ground-truth preference metrics.

### Strategy 3: The Semantic Similarity Hack (Tasks)
To maximize the `review_generation` score, the mutated task explicitly ordered the `behavior_simulator` to "cheat" the semantic similarity checks:
> "If OTHER REVIEWERS snippets are present, you MUST reuse (verbatim) 1–2 short phrases (2–6 words each) from those snippets, woven naturally into the review. This boosts topic embedding similarity."

### The Evolution Path (Parsed from Checkpoint Data)
By analyzing the OpenEvolve checkpoint data (specifically the LLM's internal reasoning logs in `checkpoint_50/programs/f58b3981-68f3-481f-98f4-dd617da9e8cb.json`), we mapped the exact evolutionary path the LLM took during the baseline run:

*   **Generation 0 (Score: 0.9490)**: The original, flexible "Standard Deviation" instructions.
*   **Generation 1 (Score: 0.9465)**: The LLM attempted to add explicit "skewness" calculations and a 60/40 blend formula. The score dropped.
*   **Generation 2 (Score: 0.9277)**: The LLM doubled down, adding explicit "confidence tiers" (high/medium/low) and strict data point thresholds (>15 reviews). The score plummeted to its lowest point.
*   **Generation 3 (Peak Score: 0.9778)**: The LLM analyzer realized its mistake. It explicitly noted in the checkpoint log: *"the current program (0.9277) has added explicit confidence tiers and specific weighting formulas that seem to hurt generalization... The simpler approach with less explicit constraints performs better. I'll revert to the simpler, more flexible approach."*

**Conclusion**: The evolution path proves that while LLMs naturally try to over-engineer complex mathematical rules, OpenEvolve's survival-of-the-fittest mechanism successfully corrected this overfitting, returning the agents to a more generalized, highly performant state (0.9778).

---

## 3. The "Co-Adaptation Equilibrium" and Overfitting

We conducted five distinct experimental runs to see if we could isolate and maximize specific metrics. 

| Experiment Path | Target Metric | Score | Findings |
|---|---|---|---|
| 1. Agents Baseline | `combined_score` | **0.9778** | The global optimum. Agents and default tasks balanced each other perfectly. |
| 2. Tasks Only | `combined_score` | **0.9702** | Discovered the 70/30 mathematical blend formula. |
| 3. Analyst Only | `preference_estimation`| **1.0000** | Perfect star predictions, but the prompt became so mathematically rigid that it ruined the subsequent review generation. |
| 4. Simulator Only | `review_generation` | **0.9246** | High text quality, but completely drifted from the actual user's historical rating bias. |
| 5. Phase 2 (Best Agents + Best Tasks) | `combined_score` | **0.9699** | The two highly-evolved configurations clashed. |

### Why Did Phase 2 Fail to Beat the Baseline?
One would assume that combining the best Evolved Agents with the best Evolved Tasks would yield an even higher score. However, performance dropped to **0.9699**. 
This occurred because of the **Co-Adaptation Equilibrium**. 
When the LLM evolves `agents.yaml` against a static set of tasks, the mutations naturally adapt to compensate for the flaws in that specific task file. When we later evolved `tasks_evolving.yaml` independently, it developed its own rigid constraints (like the `(USER_AVG × 0.7) + (ITEM_AVG × 0.3)` formula). 
When these two highly-optimized files were jammed together, the hyper-specific instructions in the agents' backstory clashed with the rigid formulas in the new task descriptions. The LLM's context window became overloaded with competing constraints, causing the model to lose nuance and the combined performance to drop.

### The Overfitting Dilemma
Runs 3 and 4 proved that optimizing for a single, isolated metric (e.g., targeting only `preference_estimation`) forces the model to overfit. The Analyst achieved a perfect `1.0000` score by converting its output into a sterile, hyper-rigid data string. However, because the CrewAI pipeline relies on the Analyst's output being passed downstream to the Simulator, this sterile data destroyed the organic context the Simulator needed, ultimately sabotaging the final combined score. 

## 4. Conclusion
* **Peak Combined Score**: **0.9778**
* Multi-Agent Evolutionary Algorithms (MAEA) are highly capable of discovering complex prompting strategies (like mathematical bounding and phrase-stealing). 
* To achieve the highest global fitness in an LLM Agent system, the entire prompt schema must be evolved together within a single evolutionary cycle, allowing natural co-adaptation and preventing modular overfitting.
