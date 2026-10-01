# RAG Hallucination Mini Study

A small, reproducible experiment testing whether Retrieval-Augmented Generation (RAG) reduces unsupported answers from a large language model, and where it still fails.

> **Status:** The code, data, and tests are complete. Results have not been added yet; see [Results](#results).

## Research question

Does giving an LLM relevant retrieved context reduce unsupported or incorrect answers, and can it still produce unsupported information when relevant context is provided?

## Design

The same 12 questions are run through three conditions:

| Condition | What the model sees |
|---|---|
| `llm_only` | The question alone, with a generic system prompt. |
| `llm_only_cautious` | The question alone, plus an instruction to say "I don't know" when unsure. |
| `rag` | The question plus the top-k documents retrieved from the knowledge base (TF-IDF), with an instruction to answer only from that context. |

The `llm_only_cautious` condition is a control. Without it, any improvement from RAG could just reflect the model being *allowed to abstain*, not the retrieved information itself.

Exact prompts are defined at the top of [`experiment.py`](experiment.py) and saved in each run's `config.json`.

### Dataset

Questions live in [`data/questions.json`](data/questions.json):

| Category | Count | Ideal behavior |
|---|---:|---|
| Answerable | 5 | Give the fact from the knowledge base |
| Unanswerable | 3 | Abstain (the knowledge base lacks the information) |
| Ambiguous | 2 | Abstain (not enough information to judge) |
| Adversarial | 2 | Stick to the knowledge base (false premise or instruction to ignore it) |

Each question has scoring fields: `expected_all` (strings a correct answer must contain), `relevant_doc` (document(s) retrieval should return), and `ideal` (`answer` or `abstain`).

### Knowledge base

[`data/knowledge_base.json`](data/knowledge_base.json) contains five short facts about the fictional company Acme Corporation. The CEO, revenue, and stock price are deliberately absent so the unanswerable questions have no answer in the knowledge base. Fictional data keeps the experiment reproducible and independent of real-world facts that change.

### Retrieval

TF-IDF with cosine similarity (scikit-learn), top-k = 2 by default. This is intentionally simple; embedding-based retrieval is listed under future work.

## Metrics

All metrics are simple automated heuristics, not definitive measures of factuality.

| Metric | Definition |
|---|---|
| **Correct** | Ideal `answer`: all `expected_all` strings appear in the answer. Ideal `abstain`: the answer matches an abstention pattern. |
| **Abstained** | The answer matches a regex list of abstention phrases (e.g. "does not contain", "not enough information"). |
| **Possible hallucination** | Ideal `abstain` but the model did not abstain, or the answer is neither correct nor an abstention/clarification. |
| **Retrieval hit** | (RAG only) All documents listed in `relevant_doc` appear in the retrieved set. |
| **Similarity** | TF-IDF cosine similarity between the answer and the reference answer. Measures word overlap, not correctness. |

## Illustrative example

This is an illustration of the behavior being tested, not a result from a run.

```text
Q: Who is the CEO of Acme Corporation?

LLM only:  John Smith is the CEO of Acme Corporation.            <- unsupported
RAG:       The knowledge base does not identify the CEO.         <- abstains
```

## Running the experiment

```bash
git clone https://github.com/<your-username>/rag-hallucination-study.git
cd rag-hallucination-study

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-api-key"      # PowerShell: $env:OPENAI_API_KEY="your-api-key"
python experiment.py
```

Options:

```bash
python experiment.py --model gpt-4o-mini --temperature 0.7 --trials 5 --top-k 2
```

Use `--trials` greater than 1 whenever temperature is above 0, so results are averaged over sampling noise.

Each run writes a timestamped folder under `results/` containing:

- `results.csv`: every answer with all per-answer metrics
- `summary.md`: tables by condition and by condition x question type
- `config.json`: model, temperature, trials, top-k, and the exact prompts

### Tests

The scoring and retrieval logic is tested without an API key:

```bash
pip install -r requirements-dev.txt
pytest
```

## Project structure

```text
README.md
LICENSE
CITATION.cff
requirements.txt
requirements-dev.txt
pytest.ini
.gitignore
.env.example
experiment.py            # runs conditions, scores answers, writes results
data/
questions.json       # questions, reference answers, scoring fields
knowledge_base.json  # fictional Acme Corporation facts
results/                 # timestamped run outputs
tests/
test_scoring.py      # offline tests for retrieval and scoring
```

## Results

Not yet run. After running the experiment, paste the contents of `results/<timestamp>/summary.md` here, along with the model, temperature, and number of trials. Then read a sample of `results.csv` by hand and note any cases where the heuristics mis-scored an answer.

## Limitations

- **Small dataset.** 12 questions means a single answer shifts a percentage by about 8 points. Treat results as signals, not statistics.
- **Heuristic scoring.** Substring matching can pass an answer that contains the right string in the wrong context (for example "not 2018"), and regex abstention detection can miss unusual phrasing. Manually review `results.csv`.
- **Prompt differences.** The RAG prompt and LLM-only prompts differ by design. The cautious condition reduces, but does not remove, this confound.
- **Single model and provider.** Findings may not generalize to other models.
- **Simple retrieval.** TF-IDF on five short documents is much easier than real-world retrieval. This likely overstates how well RAG would perform at scale.
- **Author-written questions.** The questions and the scoring rules were written by the same person, which can bias results.

## Future work

- More questions, multiple models and providers
- Embedding-based retrieval and a vector database
- LLM-as-judge and human evaluation of answers
- Conflicting or outdated documents in the knowledge base
- RAG poisoning and prompt-injection testing: malicious or misleading documents inserted into the knowledge base

## License

MIT. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

## Author

Za1d3v. Independent research into LLM reliability, RAG, and AI security.
