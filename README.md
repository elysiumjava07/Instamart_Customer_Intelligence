# Instamart Customer Intelligence & Support Copilot

## Overview
Capstone repository (synthetic data, INR). Parts A-D are built in order; each consumes the previous Part's outputs.

## Reproduce
1. `python -m venv .venv && source .venv/bin/activate`   (Windows: `.venv\Scripts\activate`)
2. `pip install pandas numpy scikit-learn joblib && pip freeze > requirements.txt`  (later: `pip install -r requirements.txt`)
3. `python generate_data.py`  (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
4. `python src/part_a_pipeline.py`   (audit, clean, write data/clean/*.csv and data/instamart.db)
5. `python src/validate_part_a.py`   (must print `ALL CHECKPOINTS PASSED`)
6. `python src/part_a_sql_eda.py`    (runs sql/queries.sql, pandas cross-check, EDA)
7. `python src/part_b_segments.py`  (RFM, K-means, PCA, writes outputs/customer_segments.csv)
8. `python src/part_b_lapse.py`      (lapse model, writes outputs/lapse_model.joblib and outputs/retention_list.csv)
9. `python src/part_c_triage.py`    (needs mock_llm.py at the repo root; writes outputs/triage_budget_demo.csv and outputs/triage_results.csv)
10. `python src/part_d_rag.py`      (chunking, retrieval, grounding evaluation with overlap 10 and 0)
11. `python src/part_d_agent.py`    (seven scenarios, stop-safely demos, crash-and-resume; writes outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/)

Shortcut: after steps 1-2, `python run_all.py` runs steps 3-11 in order and saves the combined printout of Parts A-D to `outputs/all_parts_output.txt`.

## Part A

### Audit (raw files, empty cells counted as missing; "N/A" and "error" kept as text)
| Table | Shape | Columns with missing values |
|---|---|---|
| customers | 1200 x 5 | age 61 (5.08%), gender 30 (2.50%) |
| products | 40 x 4 | none |
| orders | 13084 x 9 | delivery_minutes 2541 (19.42%) |

| Audit figure | Value |
|---|---|
| raw order rows | 13,084 |
| exact duplicate rows | 119 |
| pending rows | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age / age outside 10-100 | 61 / 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings / blank gender | 10 / 30 |

### Cleaning rules and how many rows or values each changed
**Customers**
1. City: strip spaces, title case. 476 cells changed, 6 cities remain.
2. Gender: first letter m/f mapped to male/female, blank to unknown (610 / 560 / 30).
3. Age: 6 impossible ages set missing; 67 missing ages (61 blank + 6) filled with the city median via groupby/transform. 0 missing, range 18-60.
4. signup_date converted to datetime.

**Orders (in this order)**
1. order_ts converted to datetime.
2. Exact duplicates removed: 119 rows.
3. Latest order_ts kept per order_id: 656 superseded pending rows removed, leaving 12,309.
4. order_amount made numeric (unparseable become missing); 112 missing amounts rebuilt as round(price_inr x quantity x (1 - discount_pct/100), 2).
5. Negative delivery_minutes set missing (105 left after de-duplication); 774 missing delivery times of completed orders filled with the median of the product category among completed orders. Cancelled orders stay missing (1,199) because they were never delivered.

Result: 11,110 completed, 1,199 cancelled, revenue of completed orders INR 2,998,668.80. `src/validate_part_a.py` asserts 14 checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL results (sql/queries.sql, run with pd.read_sql_query)
| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 three-table join | 30 rows, sum INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average of INR 2,552.06 |
| Q6 window ranks | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows with RANK different from ROW_NUMBER |
| Q7 running total | 183 daily rows, final cumulative INR 2,998,668.80 |

Ties in total spend give equal customers the same RANK, while ROW_NUMBER still numbers them one after another, so the two columns differ on the later tied row and on every row after it.

### pandas cross-check
The pivot_table (margins=True) grand total is INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory statistics (completed order_amount)
Mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 / 645.30, outliers below 0, above 897.
Correlation matrix (order_amount, discount_pct, quantity, delivery_minutes): order_amount-discount_pct -0.095, order_amount-quantity +0.087, all others within 0.013 of zero.

- **What is this data about?** Quick-commerce orders: one row per order with product, quantity, discount, amount and delivery time.
- **Distributions?** Amounts are strongly right-skewed (mean about twice the median) because cheap groceries dominate.
- **Outliers?** 897 orders above 645.30, mostly electronics; they are real expensive purchases, not errors, so they were kept.
- **Relationships?** The strongest pair is order_amount with discount_pct at -0.095, which is tiny; a correlation of this size does not establish causation.

## Part B

**Leakage statement.** No encoder, scaler or resampler saw test rows (preprocessing sits inside a scikit-learn Pipeline fitted on training rows only; over-sampling touched the training set only). No order dated 2025-09-01 or later entered the lapse features, and the segment labels from the RFM step were not used as features.

### RFM (snapshot 2025-10-01 00:00:00, completed orders only)
| Item | Value |
|---|---|
| customers | 1,175 |
| recency (days) median / min / max | 19 / 0 / 180 |
| frequency mean / max | 9.46 / 33 |
| monetary mean / max (INR) | 2,552.06 / 14,766.55 |

### K-means (StandardScaler, n_init=10, random_state=42)
| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4.** The inertia drops sharply up to K=4 (1268 to 743) and then flattens, so the elbow is at 4; K=4 also has the highest silhouette (0.526); and four segments are few enough for a retention team to run a separate campaign for each. Inertia always falls as K grows (K=8 is lower than K=2) because more centres can only sit closer to the points; with K equal to the number of customers it would reach zero, so inertia alone never picks K and must be read together with the elbow, silhouette and interpretability.

### Segments (cluster numbers are only labels with no ordering)
| Segment | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
|---|---|---|---|---|---|
| Premium Big Spenders (is_premium) | 138 | 24.12 | 6.77 | 7,129.70 | VIP treatment: early access to electronics, priority support |
| Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Reward loyalty with subscription bundles and streak offers |
| Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Grocery-bundle nudges and small coupons to raise frequency |
| Lapsing / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back offer before they are gone for good |

PCA on the scaled RFM matrix: explained_variance_ratio_ = 0.634 and 0.218, total 0.852.

### Lapse problem
Cutoff 2025-09-01; cohort = customers with a completed order before it: 1,156 customers, 443 lapsed (38.32%). Features (orders before the cutoff only): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days. Split (test_size=0.2, stratify=y, random_state=42): 924 train / 232 test. Baseline "never lapsed": accuracy 61.6%, lapsed-class F1 0.

Preprocessing: StandardScaler on numeric columns and OneHotEncoder(drop="first") on city, gender and favourite category, both inside the Pipeline. No feature column has a missing value (asserted in code). Scaler mean of frequency is 8.115 against 8.029 for the full cohort; they differ because the scaler only sees the 924 training rows, so no test information leaks into it.

### Tuning and test results (lapsed class)
| Model | Best parameters | Best CV F1 | Test precision | Test recall | Test F1 | Confusion [[TN, FP],[FN, TP]] |
|---|---|---|---|---|---|---|
| Baseline | n/a | n/a | 0 | 0 | 0 | n/a |
| Logistic regression | C=100 | 0.651 | 0.644 | 0.652 | 0.648 | [[111, 32],[31, 58]] |
| Random forest | max_depth=8, min_samples_leaf=10 | 0.681 | 0.659 | 0.652 | 0.655 | [[113, 30],[31, 58]] |

**Selected: random forest** (picked on cross-validated F1 from the training data, test F1 0.655, above the 0.55 requirement). Missing a lapsing customer loses all their future orders, while wrongly sending a retention offer to a loyal customer costs only a small discount, so recall matters at least as much as precision. The forest is slightly better on CV F1 and test precision and equal on recall, so it is the safer choice; if the team wants to catch even more lapsing customers, the balanced model below trades precision for recall.

### Class-balance experiment (logistic regression, training set over-sampled to 570 / 570)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved most (0.65 to 0.85) while precision fell slightly (0.64 to 0.59): the retention team would reach many more lapsing customers but contact roughly 53 loyal ones instead of 32 per 232 customers, which is acceptable when a retention offer is cheap.

### Interpretation
Top five forest features: frequency 0.280, monetary 0.271, recency 0.115, average delivery minutes 0.061, cancellation rate 0.052 (customers who order often and spend a lot rarely lapse; slow deliveries and cancellations add risk).

Logistic-regression coefficients (standardised features): frequency -1.698 (frequent buyers rarely lapse), recency +0.040 (a longer gap since the last order raises lapse risk), tenure_days -0.050 (longer-standing customers lapse slightly less).

The selected model was saved with joblib.dump to outputs/lapse_model.joblib; the reloaded model reproduces the test predictions exactly. Of the 232 test customers, 40 are in the Premium segment; the top 10 by predicted lapse probability are in outputs/retention_list.csv and 1 of them truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes September, the window that defines the lapse label.

## Part C

All model calls go through the supplied offline `mock_llm.py`. Tickets are processed in file order, `mock_llm.reset()` is called before every full run, and `tickets.csv` is read with `keep_default_na=False`.

### Guardrails and redaction
- Empty or whitespace-only text gives `skipped_empty`; `est_tokens(text) > 400` gives `flagged_long`. Neither is sent to the model. Found: 4 empty tickets and 4 long tickets (734 estimated tokens each).
- Redaction order: email, phone, order id (the three regular expressions from the brief). The user message sent to the model is exactly the redacted text.

| Type | Substitutions |
|---|---|
| [EMAIL] | 55 |
| [PHONE] | 91 |
| [ORDER_ID] | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches. The code also asserts, after every run, that none of the three patterns occurs in any message actually sent to the model.

### Prompts
Zero-shot (38 estimated tokens) and few-shot (137 estimated tokens, an `EXAMPLES:` line with four labelled examples including a fraud one). Both contain CLASSIFY, list billing, delivery, technical and fraud, and demand a JSON-only reply.

### Retries
Every call goes through `call_with_retry`: at most 3 retries per ticket, wait = 0.05 x 2^n x uniform(0.9, 1.15) seconds, one log line per retry (ticket id, retry number, wait). With this mock every retry is retry number 0, so waits lie between 0.045 and 0.0575 s. A full run over the 156 eligible tickets hit exactly 19 TransientLLMErrors, all recovered; no ticket ended with a permanent error.

### Output validation
JSON with an allowed category gives `classified_json`. Non-JSON replies are searched with `re.findall(r"[a-z]+", text.lower())` and accepted as `classified_keyword` only if exactly one allowed category word appears. Anything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to a human; nothing is guessed.

| Status (full run, either prompt) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot
| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 10,336 | 70.3 |
| Few-shot | 147 | 100% (147/147) | 25,780 | 175.4 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

All 21 zero-shot errors go to billing: 11 fraud tickets and 10 technical tickets. The few-shot prompt costs about 105 more tokens per classified ticket (175.4 against 70.3, roughly 2.5 times the total) and buys 14.3 percentage points of accuracy. That is worth paying whenever a mistake is expensive: a fraud ticket labelled billing misses the security team's 2-hour target, so for this workload the extra tokens pay for themselves; for a low-stakes bulk task the cheaper prompt would be enough.

### Budget demo and routing
With the zero-shot prompt and TOKEN_BUDGET = 6000, the run stopped calling the model after 6,036 tokens: 65 tickets ended `budget_exhausted` (79 classified_json, 7 classified_keyword, 5 flagged_invalid_output, 4 skipped_empty, 4 flagged_long); saved as `outputs/triage_budget_demo.csv`.

`outputs/triage_results.csv` (164 rows) comes from a few-shot run with TOKEN_BUDGET = 100000 (25,780 tokens used, no ticket exhausted). Routing: skipped_empty, flagged_long, flagged_invalid_output and budget_exhausted go to human_review (NORMAL); otherwise fraud goes to security_team (HIGH); otherwise premium customers go to premium_desk (HIGH); the rest go to support_queue (NORMAL). A customer missing from the segment file counts as not premium.

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

**Rule-validated versus human.** 147 tickets (135 JSON + 12 keyword) passed the output rules and were routed automatically; "validated" means the reply had a valid allowed category, not that it is guaranteed correct (the zero-shot run shows how wrong a valid answer can be). 17 tickets went to human review: 4 empty, 4 too long and 9 whose model output was invalid.

## Part D

No network call or API key is used; all model calls go through `mock_llm.py`.

### Chunking and retrieval
Each policy file is split into 40-word chunks with 10 words of overlap (stride 30), stopping after the first chunk that reaches the last word. Result: 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). Retrieval uses `TfidfVectorizer()` fitted on the chunk texts, `cosine_similarity` against the transformed query, and `retrieve(query, k=3)` returns dictionaries with source, score and text. For "refund policy late delivery" all three top chunks come from refund_policy.txt.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, informative words count more. A limitation is that TF-IDF only matches the same words: a question that says "money back" will not match a chunk that says "refund", even though the meaning is identical.

### Augment and answer
`answer(question)` retrieves three chunks and sends two messages to the mock: a system message containing POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) and a user message `CONTEXT:`, each chunk prefixed with `[<source file>]` on its own line, then `QUESTION:` and the question. It returns the model text and the sources.

### Evaluation (14 questions: 12 answerable, 2 not covered)
| Chunking | Chunks | Expected doc in top 3 (of 12) | Answer contains expected fact (of 12) | Fallback "I could not find" (of 2) |
|---|---|---|---|---|
| 40 words, overlap 10 | 15 | 12 | 12 | 2 |
| 40 words, overlap 0 | 13 | 12 | 10 | 2 |

With overlap 0 two questions break. "What delivery time makes a completed order eligible for a full refund?" returns "receives a full refund of the order amount." without the key phrase "more than 45 minutes", because the chunk boundary falls inside that sentence and separates the condition from its consequence. "How soon is money returned when a payment fails but is deducted?" returns only the first half of its sentence, missing "within 48 hours", for the same reason. Overlap keeps the full sentence together in at least one chunk.

### Tools
Five tools with names, descriptions and parameter specs in `TOOLS_SPEC` and a `TOOL_MAP` from name to function: `get_order` (reads data/instamart.db), `search_policy` (returns retrieve output under `results`), `calculate_refund` (100% of the order amount if delivery is more than 45 minutes, else 0), `issue_refund` and `escalate_to_human` (each append a timestamped JSON line to outputs/refund_ledger.jsonl). A refund above INR 1000 is escalated, never issued automatically.

### ReAct loop and the seven scenarios
`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` calls the model, executes the requested tool, appends the assistant and tool messages and loops until the reply has no tool call. The status is the word before the colon in the final answer. Every run is appended to outputs/agent_runs.jsonl with its steps.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 914 |
| S2 | REFUND | 135.00 | 5 | 914 |
| S3 | ESCALATE | 1,999.00 | 5 | 901 |
| S4 | DENIED | 0.00 | 4 | 631 |
| S5 | DENIED | 0.00 | 3 | 371 |
| S6 | NOT_FOUND | 0.00 | 2 | 120 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,805 |

All refund amounts equal expected_refund_inr. S7 completes all six model calls under the default token_limit of 6000 and then stops.

### Stopping safely
| Demonstration | Status | Detail |
|---|---|---|
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT | stopped after model call 3 at 388 tokens; the requested tool was not executed; 2 tools had run |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT | stopped before the first model call (0 calls, 0 steps) |

Every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`.

### Crash and resume
A checkpoint (run id, scenario id, full message list, tokens used, step number, executed tools, steps) is written to `state/<run id>.json` after every tool observation. For S1 a simulated crash was raised right after the second tool execution; `resume_agent("demo-S1-crash")` loaded the checkpoint and finished the run. Tools executed across the crash and the resume, in order: `get_order`, `search_policy` | crash | `calculate_refund`, `issue_refund`. Four executions, none repeated; final status REFUND with INR 128.25. The two stop-safely demos and this run are logged in outputs/agent_demos.jsonl; their ledger lines carry the run id, so the crash demo's S1 refund is distinguishable from the main S1 run.

### ReAct versus plan-and-execute
In my measurements the ReAct agent needs 5 model calls and 914 tokens for S1, 4 calls for S4 (631 tokens), 3 calls for S5 (371) and 2 calls for S6 (120). Each call re-sends the whole growing message history, so cost grows faster than the number of steps. A plan-and-execute design (which I did not build, so these are estimates) would ask for the whole step list once, run the tools without the model, and need one more call for the final answer: about 2 calls for S1 and clearly fewer tokens. The price is rigidity. If get_order returns "order not found" (S6) or a cancelled order (S5), ReAct stops after 2 or 3 calls because it reads each observation before choosing the next step. A fixed plan would keep executing steps built on a missing amount unless every step is guarded, and a replan would cost an extra call. For this refund workflow I would keep ReAct: the next step genuinely depends on what was observed (status, delivery minutes, the INR 1000 escalation threshold), and a step that moves money should be chosen after seeing the data, not before. Three to five extra calls per request are cheap next to a wrong refund. A sensible improvement is to cut tokens by sending the model only a short summary of each observation.

## Limitations and Next Steps
- The Premium flag is a post-hoc whole-window label (its RFM includes September, which defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation.
- All data is synthetic and the language model is an offline keyword mock, so classification accuracy and agent behaviour say nothing about a real hosted model.
- Cleaning imputes with medians (ages by city, delivery times by category); this keeps the checkpoints reproducible but shrinks real variation.
- The lapse model has modest test F1 (about 0.65) and uses one train/test split; a time-based split and a calibrated threshold are next steps.
- TF-IDF retrieval matches words, not meaning; embeddings or query rewriting would handle paraphrases.
- The refund tools have no idempotency key: the demo runs issue a second S1 refund to the ledger, tagged only by run id. A real system must reject a second refund for the same order.
- Token counts are the mock's word-based estimate, not billing tokens.
