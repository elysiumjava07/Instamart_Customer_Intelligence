"""Part C: support-ticket triage with guardrails. Run (repo root): python src/part_c_triage.py"""
import json, os, random, re, sys, time
import pandas as pd
sys.path.insert(0, os.getcwd())
import mock_llm
from mock_llm import TransientLLMError, est_tokens

random.seed(42)                      # makes the retry jitter reproducible
MAX_INPUT_TOKENS, MAX_RETRIES, BASE_DELAY = 400, 3, 0.05
CATS = ["billing", "delivery", "technical", "fraud"]
PATTERNS = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            (re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            (re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ("You are a support ticket router. CLASSIFY the ticket into exactly one category: "
             "billing, delivery, technical or fraud. Reply with JSON only, in the form "
             '{"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance. This is fraud. -> {"category": "fraud", "reason": "unauthorized account use"}
Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: My order [ORDER_ID] is very late and the rider is not picking up calls. -> {"category": "delivery", "reason": "late delivery"}"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

def redact(text):
    counts = {}
    for pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        counts[ph] = n
    return text, counts

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
sent_messages = []                    # every message list that reaches the model, checked at the end

def call_with_retry(messages, tid, stats):
    n = 0
    while True:
        try:
            sent_messages.append(messages)
            return mock_llm.chat(messages)
        except TransientLLMError:
            stats["transient"] += 1
            if n >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"  retry log: ticket={tid} retry={n} wait={wait:.4f}s")
            time.sleep(wait)
            n += 1

def parse(content):
    """Return (status, category). Never guesses."""
    try:
        d = json.loads(content)
        if isinstance(d, dict) and d.get("category") in CATS:
            return "classified_json", d["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        found = [w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATS]
        if len(set(found)) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None

def run(system_prompt, budget=None):
    mock_llm.reset(); sent_messages.clear()
    stats, total, rows = {"transient": 0}, 0, []
    for _, t in tickets.iterrows():
        row = dict(ticket_id=t.ticket_id, customer_id=t.customer_id, true_category=t.true_category,
                   status=None, category=None, tokens_used=0)
        text = t.text
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif budget is not None and total >= budget:
            row["status"] = "budget_exhausted"
        else:
            clean, _ = redact(text)
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            resp = call_with_retry(msgs, t.ticket_id, stats)
            used = resp.usage.total_tokens
            total += used; row["tokens_used"] = used
            row["status"], row["category"] = parse(resp.choices[0].message.content)
        rows.append(row)
    # guardrail assertion: no PII in anything sent to the model
    for m in sent_messages:
        for part in m:
            for pat, _ in PATTERNS:
                assert not pat.search(part["content"]), "PII reached the model"
    return pd.DataFrame(rows), total, stats

def route(df):
    out = df.copy()
    out["segment_name"] = out.customer_id.map(seg["segment_name"]).fillna("")
    prem = out.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
    human = out.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    fraud = ~human & (out.category == "fraud")
    pdesk = ~human & ~fraud & prem
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[pdesk, ["route", "priority"]] = ["premium_desk", "HIGH"]
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    out["category"] = out.category.fillna("")
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

# ---------------- tasks 1-2: guardrail counts and redaction ----------------
tok = tickets.text.map(est_tokens)
print("empty:", int((~tickets.text.str.strip().astype(bool)).sum()), "| long:", int((tok > MAX_INPUT_TOKENS).sum()),
      "| long token estimate:", sorted(set(tok[tok > MAX_INPUT_TOKENS])))
tot = {"[EMAIL]": 0, "[PHONE]": 0, "[ORDER_ID]": 0}; red_texts = []
for txt in tickets.text:
    r, c = redact(txt); red_texts.append(r)
    for k in tot: tot[k] += c[k]
print("redactions:", tot, "total", sum(tot.values()))
print("left after redaction:", sum(len(p.findall(r)) for r in red_texts for p, _ in PATTERNS))
print("system prompt tokens: zero-shot", est_tokens(ZERO_SHOT), "few-shot", est_tokens(FEW_SHOT))

# ---------------- tasks 4-6: zero-shot and few-shot, no budget ----------------
summary = {}
for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
    print(f"\n=== {name} run ===")
    df, total, stats = run(prompt)
    cl = df[df.status.isin(["classified_json", "classified_keyword"])]
    acc = (cl.category == cl.true_category).mean()
    summary[name] = df
    print("status counts:", df.status.value_counts().to_dict())
    print(f"TransientLLMErrors={stats['transient']}  classified={len(cl)}  accuracy={acc*100:.1f}% ({int((cl.category==cl.true_category).sum())}/{len(cl)})")
    print(f"total tokens={total}  tokens/classified ticket={total/len(cl):.1f}")
    if name == "zero-shot":
        print(pd.crosstab(cl.true_category, cl.category, rownames=["true"], colnames=["predicted"]).to_string())

# ---------------- task 7a: budget demo (zero-shot) ----------------
BUDGET_DEMO = 6000
print(f"\n=== budget demo, TOKEN_BUDGET={BUDGET_DEMO} ===")
df, total, stats = run(ZERO_SHOT, budget=BUDGET_DEMO)
route(df).to_csv("outputs/triage_budget_demo.csv", index=False)
print("status counts:", df.status.value_counts().to_dict(), "| tokens used:", total)

# ---------------- task 7b: final run (few-shot, budget 100000) ----------------
print("\n=== final few-shot run, TOKEN_BUDGET=100000 ===")
df, total, stats = run(FEW_SHOT, budget=100000)
res = route(df); res.to_csv("outputs/triage_results.csv", index=False)
print("rows:", len(res), "| status:", res.status.value_counts().to_dict(), "| tokens:", total)
print(res.groupby(["route", "priority"]).size().to_string())
