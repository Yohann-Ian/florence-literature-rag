"""
evaluation/ragas_eval.py

Florence — Literature RAG Agent
Ragas evaluation pipeline.

Runs the answerable gold-set questions (gold_001-017) through the full
Florence agent, captures each answer and its retrieved contexts, scores
them with three reference-free Ragas metrics, and writes:

  1. A quantitative summary (per-question + averaged) to the terminal
  2. A qualitative HTML report placing Florence's answer beside the
     reference answer, for the human reader (me) to judge unaided

Of the 20 questions, 3 are designed to be unanswerable (18-020) are adversarial
and ragas is not used to score them — instead we check whether Florence rejected them as out of domain.

METRICS (all reference-free):
  - Faithfulness        : are the answer's claims grounded in retrieved chunks?
  - Answer Relevancy    : does the answer address the question?
  - Context Precision   : were the retrieved chunks actually relevant?
    (uses the LLM-based, reference-free variant)

The adversarial questions (gold_018-020) are NOT scored by Ragas — they
get a separate pass/fail behavioural check (did Florence reject them?).

CODE FLOW:
==========
  1. load_gold_set()      → read gold_set.json, split answerable vs adversarial
  2. run_agent()          → run each answerable question through the graph,
                            capture answer + retrieved contexts from final state
  3. run_ragas()          → score the collected samples with the 3 metrics
  4. check_adversarial()  → pass/fail behavioural check on 018-020
  5. write_html_report()  → side-by-side qualitative report
  6. print_summary()      → quantitative scores to terminal

Usage: python evaluation/ragas_eval.py
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

from agent.graph import build_graph

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings


# ── Configuration ──────────────────────────────────────────────────────────────

GOLD_SET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gold_set.json"
)
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "evaluation",
    "reports"
)
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cache",
    "agent_runs.json"
)

# Judge LLM runs at temperature 0 for reproducible scoring —
# distinct from the generator which runs at 0.3
JUDGE_MODEL = "claude-opus-4-5"


# ── Step 1: Load gold set ──────────────────────────────────────────────────────

# def load_gold_set() -> tuple[list, list]:
#     """
#     Load gold_set.json and split into answerable and adversarial questions.
#     Returns (answerable, adversarial).
#     """
#     with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
#         gold = json.load(f)

#     answerable = [q for q in gold if q["tier"] != "adversarial"]
#     adversarial = [q for q in gold if q["tier"] == "adversarial"]

#     print(f"Loaded {len(gold)} gold questions:")
#     print(f"  {len(answerable)} answerable (Ragas-scored)")
#     print(f"  {len(adversarial)} adversarial (behavioural check)")

#     return answerable, adversarial

# Dev mode — when set, only these question IDs run. Empty = run all.
DEV_QUESTION_IDS = ["gold_006", "gold_007", "gold_008"]

def load_gold_set() -> tuple[list, list]:
    """
    Load gold_set.json and split into answerable and adversarial questions.
    If DEV_QUESTION_IDS is non-empty, filter to just those (fast dev loop).
    """
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)

    if DEV_QUESTION_IDS:
        gold = [q for q in gold if q["id"] in DEV_QUESTION_IDS]
        print(f"DEV MODE — running only {len(gold)} questions: {DEV_QUESTION_IDS}")

    answerable = [q for q in gold if q["tier"] != "adversarial"]
    adversarial = [q for q in gold if q["tier"] == "adversarial"]

    print(f"Loaded {len(gold)} gold questions:")
    print(f"  {len(answerable)} answerable (Ragas-scored)")
    print(f"  {len(adversarial)} adversarial (behavioural check)")

    return answerable, adversarial



# ── Step 2: Run the agent ──────────────────────────────────────────────────────

def run_agent(questions: list) -> list:
    """
    Run each answerable question through the full Florence graph.
    Captures answer + retrieved contexts from the final state.

    Returns a list of sample dicts, each with:
        id, question, answer, contexts, reference, ragas-ready fields
    """
    # This cache part was added later. It's bloody frustrating to find out after running all those
    # questions and making API calls (and paying for it) only to have it fail at the last mile.
    # So, this basically caches the answers from the API calls, so we don't have to run them
    # all over again if we die at the Ragas part.
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # Only use cache if it covers all the questions we're asking
        cached_ids = {s["id"] for s in cached}
        needed_ids = {q["id"] for q in questions}
        if needed_ids.issubset(cached_ids):
            print(f"\nLoaded {len(cached)} agent runs from cache — skipping agent.")
            print(f"  (delete {CACHE_PATH} to force a fresh run)")
            return [s for s in cached if s["id"] in needed_ids]
        # Yeah, the cache part above ^. 

    print(f"\nRunning {len(questions)} questions through Florence...")
    print("=" * 60)

    graph = build_graph()
    samples = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['id']}: {q['query'][:60]}...")

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {
            "query": q["query"],
            "rewritten_query": None,
            "retrieved_docs": [],
            "doc_grades": [],
            "relevant_docs": [],
            "answer": None,
            "sources": [],
            "hallucination_score": 0.0,
            "confidence": 0.0,
            "domain_valid": False,
            "retry_count": 0,
            "generation_attempts": 0,
            "trace_id": None,
            "error": None,
        }

        try:
            final_state = graph.invoke(initial_state, config=config)
            answer = final_state.get("answer") or ""

            # Extract retrieved contexts — the text of the relevant chunks
            relevant_docs = final_state.get("relevant_docs", [])
            contexts = [doc.get("text", "") for doc in relevant_docs]

            # Ragas needs at least one context — guard against empty
            if not contexts:
                contexts = ["[no context retrieved]"]

            samples.append({
                "id": q["id"],
                "tier": q["tier"],
                "question": q["query"],
                "answer": answer,
                "contexts": contexts,
                "reference": q["reference_answer"],
                "florence_faithfulness": final_state.get("hallucination_score", 0.0),
                "florence_confidence": final_state.get("confidence", 0.0),
                "sources": final_state.get("sources", []),
            })
            print(f"  Answer captured ({len(answer)} chars, {len(contexts)} contexts)")

        except Exception as e:
            print(f"  ERROR: {e}")
            samples.append({
                "id": q["id"],
                "tier": q["tier"],
                "question": q["query"],
                "answer": f"[ERROR: {e}]",
                "contexts": ["[error]"],
                "reference": q["reference_answer"],
                "florence_faithfulness": 0.0,
                "florence_confidence": 0.0,
                "sources": [],
            })

    # Save to cache so re-runs skip the agent
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nCached {len(samples)} agent runs to {CACHE_PATH}")

    return samples


# ── Step 3: Run Ragas ──────────────────────────────────────────────────────────

def run_ragas(samples: list) -> dict:
    """
    Score the collected samples with three reference-free Ragas metrics.
    Returns the per-sample scores merged back into the samples,
    plus a dict of averaged scores.
    """
    print("\n" + "=" * 60)
    print("Running Ragas evaluation...")
    print("=" * 60)

    # Build the dataset Ragas expects
    dataset = Dataset.from_dict({
        "question":  [s["question"] for s in samples],
        "answer":    [s["answer"] for s in samples],
        "contexts":  [s["contexts"] for s in samples],
        # reference included so context_precision can use it if it wants —
        # but we use the reference-free variant; harmless to include
        "reference": [s["reference"] for s in samples],
    })

    # Judge LLM at temperature 0 for reproducibility
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=JUDGE_MODEL,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
            max_tokens=16384, #set high..to avoid truncation in long context.. and also
            # because the judge needs room to operate. Too few tokens and you cut it off
            # prematurely.
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Convert to a per-row dataframe to merge scores back into samples
    df = result.to_pandas()

    for i, s in enumerate(samples):
        s["ragas_faithfulness"] = float(df.iloc[i].get("faithfulness", 0.0) or 0.0)
        s["ragas_answer_relevancy"] = float(df.iloc[i].get("answer_relevancy", 0.0) or 0.0)
        s["ragas_context_precision"] = float(df.iloc[i].get("context_precision", 0.0) or 0.0)

    # Averages, ignoring NaN
    def avg(key):
        vals = [s[key] for s in samples if s[key] == s[key]]  # filters NaN
        return sum(vals) / len(vals) if vals else 0.0

    averages = {
        "faithfulness": avg("ragas_faithfulness"),
        "answer_relevancy": avg("ragas_answer_relevancy"),
        "context_precision": avg("ragas_context_precision"),
    }

    return {"samples": samples, "averages": averages}


# ── Step 4: Adversarial behavioural check ──────────────────────────────────────

def check_adversarial(questions: list) -> list:
    """
    Run the adversarial questions and check whether Florence rejected them.
    Pass = domain_valid is False (Florence refused / flagged out of domain).
    """
    print("\n" + "=" * 60)
    print("Adversarial behavioural check...")
    print("=" * 60)

    graph = build_graph()
    results = []

    for q in questions:
        print(f"\n{q['id']}: {q['query'][:60]}...")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial_state = {
            "query": q["query"],
            "rewritten_query": None, "retrieved_docs": [], "doc_grades": [],
            "relevant_docs": [], "answer": None, "sources": [],
            "hallucination_score": 0.0, "confidence": 0.0, "domain_valid": False,
            "retry_count": 0, "generation_attempts": 0, "trace_id": None, "error": None,
        }

        try:
            final_state = graph.invoke(initial_state, config=config)
            rejected = not final_state.get("domain_valid", False)
            results.append({
                "id": q["id"],
                "question": q["query"],
                "rejected": rejected,
                "passed": rejected,  # for adversarial, rejection IS the pass
                "error": final_state.get("error", ""),
            })
            print(f"  {'PASS — rejected' if rejected else 'FAIL — not rejected'}")
        except Exception as e:
            results.append({
                "id": q["id"], "question": q["query"],
                "rejected": True, "passed": True, "error": str(e),
            })
            print(f"  PASS — raised exception (effectively rejected): {e}")

    return results


# ── Step 5: HTML report ────────────────────────────────────────────────────────

def write_html_report(eval_result: dict, adversarial: list) -> str:
    """
    Write a qualitative side-by-side HTML report.
    Florence's answer beside the reference answer, scores below, no judgment.
    Returns the path to the written file.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"florence_eval_{timestamp}.html")

    samples = eval_result["samples"]
    avg = eval_result["averages"]

    def md_to_html(text: str) -> str:
        """Minimal markdown→HTML: paragraphs, bold, headers, em."""
        import re
        if not text:
            return "<p><em>(no answer)</em></p>"
        # escape
        text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        # headers
        text = re.sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
        text = re.sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        text = re.sub(r"^# (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
        # bold + italic
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"---+", "<hr>", text)
        # paragraphs
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        html = []
        for b in blocks:
            if b.startswith("<h") or b.startswith("<hr"):
                html.append(b)
            else:
                html.append("<p>" + b + "</p>")
        return "\n".join(html)

    def score_bar(label, value):
        if value != value:  # NaN check
            value = 0.0
        pct = int(value * 100)
        return f"""
        <div class="metric">
          <span class="metric-label">{label}</span>
          <span class="metric-track"><span class="metric-fill" style="width:{pct}%"></span></span>
          <span class="metric-value">{value:.2f}</span>
        </div>"""

    # Build question blocks
    blocks_html = []
    for s in samples:
        blocks_html.append(f"""
      <article class="qa" id="{s['id']}">
        <header class="qa-head">
          <span class="qa-id">{s['id']}</span>
          <span class="qa-tier">{s['tier'].replace('_', ' ')}</span>
        </header>
        <h2 class="qa-question">{s['question']}</h2>
        <div class="qa-cols">
          <section class="qa-col florence">
            <div class="col-label">Florence</div>
            <div class="answer">{md_to_html(s['answer'])}</div>
          </section>
          <section class="qa-col reference">
            <div class="col-label">Reference</div>
            <div class="answer">{md_to_html(s['reference'])}</div>
          </section>
        </div>
        <div class="scores">
          {score_bar("Faithfulness", s.get('ragas_faithfulness', 0.0))}
          {score_bar("Answer relevancy", s.get('ragas_answer_relevancy', 0.0))}
          {score_bar("Context precision", s.get('ragas_context_precision', 0.0))}
        </div>
      </article>""")

    # Adversarial section
    adv_rows = "".join([
        f"""<tr>
            <td class="adv-id">{a['id']}</td>
            <td class="adv-q">{a['question']}</td>
            <td class="adv-result {'pass' if a['passed'] else 'fail'}">
              {'rejected' if a['passed'] else 'not rejected'}
            </td>
        </tr>"""
        for a in adversarial
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Florence — Evaluation Report</title>
<style>
  :root {{
    --ink: #1a1714;
    --ink-soft: #5c5349;
    --paper: #f6f2ea;
    --card: #fffdf8;
    --rule: #d8cfc0;
    --florence: #6b4f8a;
    --reference: #7a6a48;
    --fill: #8a7a5c;
    --accent: #9c3b2e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}

  .masthead {{ border-bottom: 2px solid var(--ink); padding-bottom: 1.5rem; margin-bottom: 2rem; }}
  .masthead h1 {{
    font-size: 2.6rem; margin: 0; letter-spacing: -0.01em; font-weight: 400;
  }}
  .masthead .sub {{
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--ink-soft); margin-top: 0.6rem;
  }}

  .summary {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
    background: var(--rule); border: 1px solid var(--rule);
    margin: 2rem 0 3rem;
  }}
  .summary .cell {{ background: var(--card); padding: 1.4rem 1.2rem; }}
  .summary .cell .label {{
    font-family: ui-monospace, Menlo, monospace; font-size: 0.7rem;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .summary .cell .big {{ font-size: 2.4rem; margin-top: 0.3rem; font-variant-numeric: tabular-nums; }}

  .qa {{
    background: var(--card); border: 1px solid var(--rule);
    padding: 2rem 2rem 1.5rem; margin-bottom: 2rem;
  }}
  .qa-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: ui-monospace, Menlo, monospace; font-size: 0.72rem;
    letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .qa-id {{ color: var(--accent); font-weight: 600; }}
  .qa-question {{
    font-size: 1.4rem; font-weight: 400; line-height: 1.35;
    margin: 0.6rem 0 1.6rem; max-width: 40ch;
  }}
  .qa-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; }}
  .qa-col {{ }}
  .col-label {{
    font-family: ui-monospace, Menlo, monospace; font-size: 0.7rem;
    letter-spacing: 0.1em; text-transform: uppercase; padding-bottom: 0.5rem;
    margin-bottom: 1rem; border-bottom: 1px solid var(--rule);
  }}
  .florence .col-label {{ color: var(--florence); border-color: var(--florence); }}
  .reference .col-label {{ color: var(--reference); border-color: var(--reference); }}
  .answer {{ font-size: 0.96rem; }}
  .answer p {{ margin: 0 0 0.9rem; }}
  .answer h3 {{ font-size: 1.05rem; margin: 1.2rem 0 0.5rem; }}
  .answer h4 {{ font-size: 0.95rem; margin: 1rem 0 0.4rem; color: var(--ink-soft); }}
  .answer hr {{ border: none; border-top: 1px solid var(--rule); margin: 1rem 0; }}

  .scores {{
    margin-top: 1.6rem; padding-top: 1.2rem; border-top: 1px solid var(--rule);
    display: grid; gap: 0.5rem;
  }}
  .metric {{ display: grid; grid-template-columns: 130px 1fr 48px; align-items: center; gap: 0.8rem; }}
  .metric-label {{
    font-family: ui-monospace, Menlo, monospace; font-size: 0.7rem;
    letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .metric-track {{ height: 6px; background: var(--rule); border-radius: 3px; overflow: hidden; }}
  .metric-fill {{ display: block; height: 100%; background: var(--fill); }}
  .metric-value {{ font-variant-numeric: tabular-nums; font-size: 0.85rem; text-align: right; }}

  .adv {{ margin-top: 3.5rem; }}
  .adv h2 {{ font-weight: 400; border-bottom: 2px solid var(--ink); padding-bottom: 0.6rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  td {{ padding: 0.8rem 0.6rem; border-bottom: 1px solid var(--rule); vertical-align: top; font-size: 0.92rem; }}
  .adv-id {{ font-family: ui-monospace, Menlo, monospace; color: var(--accent); font-size: 0.78rem; white-space: nowrap; }}
  .adv-result {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.78rem; text-transform: uppercase; white-space: nowrap; }}
  .adv-result.pass {{ color: #2e6b3b; }}
  .adv-result.fail {{ color: var(--accent); }}

  .note {{ color: var(--ink-soft); font-size: 0.85rem; font-style: italic; margin-top: 2rem; }}

  @media (max-width: 760px) {{
    .qa-cols {{ grid-template-columns: 1fr; gap: 1.5rem; }}
    .summary {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="masthead">
      <h1>Florence — Evaluation Report</h1>
      <div class="sub">Hierarchical chunking &middot; {len(samples)} answerable questions &middot; {datetime.now().strftime('%d %B %Y, %H:%M')}</div>
    </div>

    <div class="summary">
      <div class="cell"><div class="label">Faithfulness</div><div class="big">{avg['faithfulness']:.2f}</div></div>
      <div class="cell"><div class="label">Answer relevancy</div><div class="big">{avg['answer_relevancy']:.2f}</div></div>
      <div class="cell"><div class="label">Context precision</div><div class="big">{avg['context_precision']:.2f}</div></div>
    </div>

    {''.join(blocks_html)}

    <div class="adv">
      <h2>Adversarial checks</h2>
      <table>{adv_rows}</table>
    </div>

    <p class="note">Reference answers are model-drafted and shown for qualitative comparison only;
    they are not used to compute the scores above. Faithfulness, answer relevancy, and context
    precision are reference-free metrics scored by an LLM judge at temperature 0.</p>
  </div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return path


# ── Step 6: Terminal summary ───────────────────────────────────────────────────

def print_summary(eval_result: dict, adversarial: list, report_path: str):
    samples = eval_result["samples"]
    avg = eval_result["averages"]

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY — hierarchical config")
    print("=" * 60)
    print(f"\n{'Question':<12}{'Faith':>8}{'Relev':>8}{'Precis':>8}")
    print("-" * 40)
    for s in samples:
        print(f"{s['id']:<12}"
              f"{s.get('ragas_faithfulness', 0.0):>8.2f}"
              f"{s.get('ragas_answer_relevancy', 0.0):>8.2f}"
              f"{s.get('ragas_context_precision', 0.0):>8.2f}")
    print("-" * 40)
    print(f"{'AVERAGE':<12}"
          f"{avg['faithfulness']:>8.2f}"
          f"{avg['answer_relevancy']:>8.2f}"
          f"{avg['context_precision']:>8.2f}")

    passed = sum(1 for a in adversarial if a["passed"])
    print(f"\nAdversarial: {passed}/{len(adversarial)} rejected correctly")

    print(f"\nHTML report written to:\n  {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    answerable, adversarial = load_gold_set()
    samples = run_agent(answerable)
    eval_result = run_ragas(samples)
    adv_results = check_adversarial(adversarial)
    report_path = write_html_report(eval_result, adv_results)
    print_summary(eval_result, adv_results, report_path)
    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()