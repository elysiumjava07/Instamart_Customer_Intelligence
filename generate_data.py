"""Run the whole Copilot pipeline (Parts A-D) in order and save one combined output log.
Usage (repo root, virtual env active): python run_all.py   -> prints and writes outputs/all_parts_output.txt"""
import os, subprocess, sys

STEPS = [("SETUP: generate data", ["generate_data.py"]),
         ("PART A: audit, clean, store", ["src/part_a_pipeline.py"]),
         ("PART A: validation", ["src/validate_part_a.py"]),
         ("PART A: SQL queries, pandas cross-check, EDA", ["src/part_a_sql_eda.py"]),
         ("PART B: RFM, K-means, segments, PCA", ["src/part_b_segments.py"]),
         ("PART B: lapse model", ["src/part_b_lapse.py"]),
         ("PART C: ticket triage", ["src/part_c_triage.py"]),
         ("PART D: policy RAG", ["src/part_d_rag.py"]),
         ("PART D: refund agent", ["src/part_d_agent.py"])]
os.makedirs("outputs", exist_ok=True)
log = []
for title, cmd in STEPS:
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    block = f"\n{'=' * 78}\n{title}   [python {' '.join(cmd)}]\n{'=' * 78}\n{r.stdout}{r.stderr}"
    print(block); log.append(block)
    if r.returncode != 0:
        print(f"STOPPED: {title} failed with exit code {r.returncode}"); break
open("outputs/all_parts_output.txt", "w", encoding="utf-8").write("".join(log))
