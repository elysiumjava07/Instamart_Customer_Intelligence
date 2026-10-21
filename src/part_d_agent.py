"""Part D tasks 5-8: tools, ReAct loop, safe stopping, resumable runs.
Run (repo root, after part_a_pipeline.py): python src/part_d_agent.py"""
import glob, json, math, os, shutil, sqlite3, sys, time
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0, os.getcwd())
import mock_llm
from part_d_rag import retrieve

DB, LEDGER, RUNS, STATE = "data/instamart.db", "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "state"
SYSTEM_AGENT = ("You are a refund AGENT. Use the tools to look up the order, check the policy, "
                "calculate the refund and issue it or escalate it. Finish with STATUS: explanation.")
CURRENT = {"run_id": None}      # lets ledger lines say which run wrote them

# ---------------- tools ----------------
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or (isinstance(row[3], float) and math.isnan(row[3])) else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": dm}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]} for h in retrieve(query, k=3)]}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "run_id": CURRENT["run_id"],
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):      return _ledger("issue_refund", order_id, refund_inr)
def escalate_to_human(order_id, refund_inr): return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props):
    return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": list(props)}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the three best matching chunks.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

# ---------------- ReAct loop ----------------
class SimulatedCrash(Exception): pass

def _dumps(x): return json.dumps(x, separators=(",", ":"))

def _checkpoint(state):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w") as f:
        json.dump(state, f)

def _finish(state, status, answer):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"]}

def _handoff(state, reason):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    return f"Handing over to a human agent: {reason}. Tools completed so far: {done}."

def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools):
    start = time.monotonic()
    CURRENT["run_id"] = state["run_id"]
    while state["llm_calls"] < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return _finish(state, "FALLBACK_TIMEOUT", _handoff(state, "time limit reached"))
        resp = mock_llm.chat(state["messages"], TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:
            return _finish(state, "FALLBACK_TOKEN_LIMIT", _handoff(state, "token limit exceeded"))
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content
            return _finish(state, text.split(":")[0].strip(), text)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": _dumps(obs)})
        state["step"] += 1
        state["executed_tools"].append(tc.name)
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        _checkpoint(state)
        if crash_after_tools and len(state["executed_tools"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")
    return _finish(state, "FALLBACK_MAX_ITERATIONS", _handoff(state, f"no final answer after {max_iterations} model calls"))

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"run-{scenario_id}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": SYSTEM_AGENT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "executed_tools": [], "steps": [],
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json")) as f:
        state = json.load(f)
    lim = state["limits"]
    return _loop(state, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"], None)

def refund_of(res):
    for s in res["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0

if __name__ == "__main__":
    for p in (LEDGER, RUNS, "outputs/agent_demos.jsonl"):
        if os.path.exists(p): os.remove(p)
    shutil.rmtree(STATE, ignore_errors=True)
    sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    print("scenario status refund expected_status expected_refund llm_calls tokens")
    for _, r in sc.iterrows():
        res = run_agent(r.message, r.scenario_id)
        with open(RUNS, "a") as f:
            f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
        amt = refund_of(res)
        exp = "FALLBACK_MAX_ITERATIONS" if r.expected_outcome == "FALLBACK" else r.expected_outcome
        ok = res["status"] == exp and abs(amt - float(r.expected_refund_inr)) < 0.01
        print(f"{r.scenario_id} {res['status']} {amt:.2f} {exp} {float(r.expected_refund_inr):.2f} {res['llm_calls']} {res['tokens_used']} {'OK' if ok else 'MISMATCH'}")
        if r.scenario_id == "S1": s1_msg = r.message

    demos = []
    d1 = run_agent(s1_msg, "S1", token_limit=200, run_id="demo-S1-token-limit")
    d2 = run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="demo-S1-timeout")
    print("\ntoken_limit=200 demo:", d1["status"], "| calls", d1["llm_calls"], "| tokens", d1["tokens_used"], "| steps", len(d1["steps"]))
    print("time_limit_seconds=0 demo:", d2["status"], "| calls", d2["llm_calls"], "| steps", len(d2["steps"]))
    print("  handoff:", d1["final_answer"])

    try:
        run_agent(s1_msg, "S1", run_id="demo-S1-crash", crash_after_tools=2)
    except SimulatedCrash as e:
        print("\ncrash:", e)
    d3 = resume_agent("demo-S1-crash")
    with open(os.path.join(STATE, "demo-S1-crash.json")) as f: ck = json.load(f)
    print("tools executed across crash and resume:", ck["executed_tools"])
    print("resumed result:", d3["status"], refund_of(d3), "| llm_calls", d3["llm_calls"], "| tokens", d3["tokens_used"])
    with open("outputs/agent_demos.jsonl", "w") as f:
        for d in (d1, d2, d3):
            f.write(json.dumps({k: d[k] for k in ("run_id", "scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| checkpoints:", len(glob.glob(f"{STATE}/*.json")))
