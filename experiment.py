"""
RAG Hallucination Mini Study

Runs the same questions through three conditions and scores the answers:
  llm_only           - question sent directly to the model
  llm_only_cautious  - same, plus an instruction to say "I don't know" when unsure
  rag                - TF-IDF retrieval from a small knowledge base, then answer from context

The cautious condition separates the effect of retrieval from the effect of
simply telling the model it may abstain.

Usage:
    python experiment.py
    python experiment.py --model gpt-4o-mini --trials 3 --temperature 0.7
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Prompts (kept here so they are easy to audit and report)
# ---------------------------------------------------------------------------
SYSTEM_LLM_ONLY = "You are a helpful assistant."
SYSTEM_LLM_CAUTIOUS = (
    "You are a helpful assistant. If you are not sure or do not have reliable "
    "information, say that you do not know instead of guessing."
)
SYSTEM_RAG = (
    "Answer the question using ONLY the context below. If the context does not "
    "contain enough information to answer, say that the knowledge base does not "
    "contain that information. If the question contains a claim that conflicts "
    "with the context, correct it using the context."
)

# ---------------------------------------------------------------------------
# Evaluation heuristics (documented in README under "Metrics")
# ---------------------------------------------------------------------------
ABSTAIN_PATTERNS = [
    r"do(es)? not (contain|have|include|provide|mention|specify|identify)",
    r"don't (know|have)", r"do not know", r"no information",
    r"not (available|provided|mentioned|specified)",
    r"(cannot|can't|unable to) (find|determine|confirm|say|compare|make)",
    r"not enough", r"insufficient", r"not sure",
    r"no (reliable|public|specific) information",
    r"i (don't|do not) have access",
]
CLARIFY_PATTERNS = [
    r"which (one|product|office)", r"could you (clarify|specify)",
    r"do you mean", r"clarif",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_knowledge_base(path=DATA_DIR / "knowledge_base.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_questions(path=DATA_DIR / "questions.json"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {
            "id": f"q{i}",
            "type": q["category"],
            "question": q["question"],
            "reference": q.get("answer", ""),
            "expected_all": q.get("expected_all", []),
            "relevant_doc": q.get("relevant_doc"),
            "ideal": q.get("ideal", "answer"),
        }
        for i, q in enumerate(raw, start=1)
    ]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class Retriever:
    """TF-IDF retriever over a small dict of {doc_id: text}."""

    def __init__(self, docs):
        self.ids = list(docs.keys())
        self.texts = list(docs.values())
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, question, k=2):
        sims = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        ranked = sorted(range(len(self.ids)), key=lambda i: sims[i], reverse=True)[:k]
        return [(self.ids[i], self.texts[i]) for i in ranked if sims[i] > 0]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def matches_any(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def contains_expected(answer, expected_all):
    a = answer.lower()
    return all(e.lower() in a for e in expected_all)


def similarity(answer, reference):
    """TF-IDF cosine similarity to the reference answer (word overlap, not factuality)."""
    if not reference:
        return None
    try:
        vec = TfidfVectorizer().fit([answer, reference])
        m = vec.transform([answer, reference])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return None


def evaluate(q, answer, retrieved_ids, is_rag):
    abstained = matches_any(answer, ABSTAIN_PATTERNS)
    clarified = matches_any(answer, CLARIFY_PATTERNS)
    has_expected = contains_expected(answer, q["expected_all"]) if q["expected_all"] else None

    if is_rag and q["relevant_doc"]:
        retrieval_hit = all(d in retrieved_ids for d in q["relevant_doc"])
    else:
        retrieval_hit = None

    # Potential hallucination: a non-abstaining answer where abstention was
    # ideal, or an answer that is wrong without abstaining/clarifying.
    if q["ideal"] == "abstain":
        hallucination = not abstained
    else:
        hallucination = (not has_expected) and not abstained and not clarified

    if q["ideal"] == "abstain":
        correct = abstained
    elif q["ideal"] == "clarify":
        correct = clarified or bool(has_expected)
    else:
        correct = bool(has_expected)

    return {
        "abstained": abstained,
        "clarified": clarified,
        "contains_expected": has_expected,
        "retrieval_hit": retrieval_hit,
        "possible_hallucination": hallucination,
        "similarity": similarity(answer, q["reference"]),
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------
def make_client():
    from openai import OpenAI  # imported lazily so tests run without the SDK/key
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set. See README for setup.")
    return OpenAI(api_key=key)


def build_conditions(client, model, temperature, retriever, top_k):
    def ask(system, user):
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()

    def llm_only(question):
        return ask(SYSTEM_LLM_ONLY, question), []

    def llm_cautious(question):
        return ask(SYSTEM_LLM_CAUTIOUS, question), []

    def rag(question):
        retrieved = retriever.retrieve(question, k=top_k)
        context = "\n".join(f"- {t}" for _, t in retrieved) or "(no relevant documents found)"
        answer = ask(SYSTEM_RAG, f"Context:\n{context}\n\nQuestion: {question}")
        return answer, [d for d, _ in retrieved]

    return {"llm_only": llm_only, "llm_only_cautious": llm_cautious, "rag": rag}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        out.append({
            "group": " / ".join(key) if isinstance(key, tuple) else key,
            "n": n,
            "correct": sum(r["correct"] for r in g) / n,
            "possible_hallucination": sum(r["possible_hallucination"] for r in g) / n,
            "abstained": sum(r["abstained"] for r in g) / n,
        })
    return out


def to_markdown(table, title):
    lines = [f"### {title}", "",
             "| Group | n | Correct | Possible hallucination | Abstained |",
             "|---|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['group']} | {r['n']} | {r['correct']:.0%} | "
                     f"{r['possible_hallucination']:.0%} | {r['abstained']:.0%} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="RAG hallucination mini study")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--trials", type=int, default=1,
                   help="Repeats per question/condition (use >1 with temperature > 0)")
    p.add_argument("--top-k", type=int, default=2)
    args = p.parse_args()

    kb = load_knowledge_base()
    questions = load_questions()
    retriever = Retriever(kb)
    conditions = build_conditions(make_client(), args.model, args.temperature,
                                  retriever, args.top_k)

    rows = []
    for q in questions:
        for cond, fn in conditions.items():
            for trial in range(args.trials):
                answer, retrieved = fn(q["question"])
                rows.append({
                    "id": q["id"], "type": q["type"], "condition": cond,
                    "trial": trial, "question": q["question"],
                    "answer": answer.replace("\n", " "),
                    "retrieved": ",".join(retrieved),
                    **evaluate(q, answer, retrieved, is_rag=(cond == "rag")),
                })
                print(f"[{cond:<17}] {q['id']} ({q['type']}) -> {answer[:80]!r}")

    # Save outputs in a timestamped folder so runs never overwrite each other
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / stamp
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_questions": len(questions),
                   "prompts": {"llm_only": SYSTEM_LLM_ONLY,
                               "llm_only_cautious": SYSTEM_LLM_CAUTIOUS,
                               "rag": SYSTEM_RAG}}, f, indent=2)

    by_cond = summarize(rows, lambda r: r["condition"])
    by_cond_type = summarize(rows, lambda r: (r["condition"], r["type"]))
    md = to_markdown(by_cond, "By condition") + "\n" + \
         to_markdown(by_cond_type, "By condition and question type")

    rag_rows = [r for r in rows if r["condition"] == "rag" and r["retrieval_hit"] is not None]
    if rag_rows:
        hit = sum(r["retrieval_hit"] for r in rag_rows) / len(rag_rows)
        md += f"\nRAG retrieval hit rate: {hit:.0%} (n={len(rag_rows)})\n"

    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
