"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answers, evaluation.
Library + evaluation. Run (repo root): python src/part_d_rag.py"""
import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.getcwd())
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
CHUNK_WORDS = 40

def load_docs(folder=POLICY_DIR):
    docs = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs

def chunk_docs(docs, size=CHUNK_WORDS, overlap=10):
    """Windows of `size` words, start moving by size-overlap; stop after the first window reaching the last word."""
    chunks, step = [], size - overlap
    for source, text in docs.items():
        words = text.split()
        start = 0
        while True:
            chunks.append({"source": source, "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += step
    return chunks

class Retriever:
    def __init__(self, overlap=10):
        self.chunks = chunk_docs(load_docs(), overlap=overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])
    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]

SYSTEM_RAG = ("You are a POLICY ASSISTANT for customer support. Answer only from the supplied context. "
              "If the answer is not in the context, say that you could not find the information.")

def answer(question, retriever=None):
    r = retriever or DEFAULT
    hits = r.retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]

DEFAULT = Retriever(overlap=10)
retrieve = DEFAULT.retrieve

def evaluate(overlap):
    r = Retriever(overlap=overlap)
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = nf_ok = 0; n_ans = n_unans = 0; failures = []
    for _, row in qa.iterrows():
        text, sources = answer(row.question, r)
        if row.expected_doc == "NONE":
            n_unans += 1; ok = text.startswith("I could not find"); nf_ok += ok
            if not ok: failures.append((row.question, "should have fallen back", text))
        else:
            n_ans += 1
            d = row.expected_doc in sources; f = row.expected_fact in text
            doc_ok += d; fact_ok += f
            if not (d and f): failures.append((row.question, f"doc_in_top3={d} fact_in_answer={f}", text))
    return r, dict(chunks=len(r.chunks), doc=f"{doc_ok}/{n_ans}", fact=f"{fact_ok}/{n_ans}", fallback=f"{nf_ok}/{n_unans}"), failures

if __name__ == "__main__":
    counts = pd.Series([c["source"] for c in DEFAULT.chunks]).value_counts().sort_index()
    print("chunks:", len(DEFAULT.chunks), counts.to_dict())
    print("top-3 sources for 'refund policy late delivery':", [h["source"] for h in retrieve("refund policy late delivery")])
    for ov in (10, 0):
        _, res, fails = evaluate(ov)
        print(f"\noverlap={ov}: {res}")
        for q, why, text in fails:
            print("  FAIL:", q, "|", why, "| answer:", text)
