# Industrial RAG Assistant

Ask questions about industrial technical manuals (PDF) and get answers that **cite the document and page** they came from. When the documents do not contain the answer, the assistant says so instead of guessing.

![Answer with citation](docs/screenshot-answer.jpg)

## Why this exists

A general-purpose LLM asked "what is the tightening torque of the impeller nut?" will happily produce a plausible number. In a maintenance context a plausible-but-invented number is dangerous. This project implements Retrieval-Augmented Generation (RAG) from its basic components (no LangChain / LlamaIndex), with the focus on **verifiability**: every answer is tied to retrievable evidence, and the quality of the system is measured, not assumed.

## Features

- Upload one or more PDFs in a Streamlit UI (processed in memory, never stored), with a switchable Italian / English interface (`src/rag_assistant/i18n.py`).
- Page-aware extraction: every passage keeps its document name and page number.
- Configurable chunking and exact FAISS vector search, with **two search methods**: by keywords (default, light) or by meaning (neural multilingual embeddings, optional).
- Answers restricted to the retrieved evidence, with citations checked against the passages that were really retrieved.
- Explicit "not enough information" answers, from a similarity threshold plus the answering engine.
- Defences against prompt injection hidden in PDFs and conflict reporting between documents.
- Retrieved passages with similarity scores are always shown, so the user can check the evidence.
- **Page preview**: the cited PDF page is rendered with the passage used highlighted, so a citation can be checked at a glance.
- Answers containing technical values (torque, pressure, temperature...) carry a reminder to verify them on the cited page.
- Four tabs: **Assistant**, **How it works** (plain-language explanation), **Architecture** (two-flow diagram, modules, technologies, design decisions) and **Evaluation** (live view of `eval/results_*.json`, including the observed errors).
- Two interchangeable answer engines: an **offline extractive engine** (no API key) and **Claude** via the Anthropic API.
- 200 automated tests that run with no network and no API key, plus a repeatable evaluation.

## Quick start

Requires Python 3.10+ (developed and tested on 3.12).

```bash
git clone <your-repo-url> industrial-rag-assistant
cd industrial-rag-assistant
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[app,dev]"
streamlit run app/streamlit_app.py
```

Upload your own PDF manuals in the sidebar and ask a question. To try the app without your own documents, upload the synthetic PDFs from `data/sample_pdfs/` and ask, for example:

- `What is the tightening torque of the impeller nut?`
- `Qual è la pressione massima di esercizio del compressore C-50?`
- `What is the maximum pressure of the pump?` (two documents disagree: both values are reported)
- `What is the warranty period of the P-200?` (not in the documents: the assistant declines)

The default **keyword search** needs no download. The optional **semantic search** downloads an embedding model (~120 MB, 8-bit quantised) from Hugging Face on first use. Behind a corporate proxy that re-signs TLS traffic and shows `CERTIFICATE_VERIFY_FAILED`, run `pip install -e ".[corp]"`: the app then uses the operating-system certificate store.

### Using Claude instead of the offline engine

Choose **Claude** in the sidebar and paste an API key (kept in memory only), or set it in the environment:

```bash
cp .env.example .env        # then edit .env; .env is git-ignored
export ANTHROPIC_API_KEY=...   # PowerShell: $env:ANTHROPIC_API_KEY="..."
```

The Claude engine is implemented and unit-tested against a fake provider, but **it has not been evaluated against the live API** in this repository's results (see Limitations).

### Tests, lint, evaluation

```bash
pytest                         # 200 tests, offline, no API key
ruff check . && ruff format --check .
python -m rag_assistant.evaluation --search keyword  --out eval/results_keyword.json
python -m rag_assistant.evaluation --search semantic --out eval/results_semantic.json   # needs the model
python scripts/make_sample_pdfs.py                            # regenerate the sample PDFs
```

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion [Once per PDF]
        A[PDF upload] --> B[pdf_loader<br/>pages + page numbers]
        B --> C[chunking<br/>overlapping chunks per page]
        C --> D[embeddings<br/>keywords or neural]
        D --> E[(FAISS index<br/>cosine similarity)]
    end
    subgraph Query [Every question]
        Q[Question] --> F[embed question]
        F --> G[retrieval<br/>top-k + scores]
        E --> G
        G --> H{generation<br/>score threshold<br/>+ injection filter}
        H -- no usable evidence --> N[Not enough information]
        H -- evidence --> I[Prompt: rules + untrusted sources]
        I --> J[LLM provider<br/>Claude or offline extractive]
        J --> K[Citation check<br/>[S1] -> document, page]
        K --> L[Answer + sources + passages]
    end
```

| Module | Responsibility |
|---|---|
| `pdf_loader.py` | PDF bytes → pages; typed errors for empty / corrupted / password-protected / text-less PDFs |
| `chunking.py` | Sliding window with overlap, cut on whitespace, never crossing a page boundary |
| `textutil.py` | Stopwords and a crude prefix stemmer shared by keyword search and the offline engine |
| `embeddings.py` | `Embedder` protocol; `LexicalEmbedder` (keywords, default), `FastEmbedEmbedder` (neural), `HashingEmbedder` (deterministic, for tests) |
| `index.py` | FAISS `IndexFlatIP` over unit vectors (inner product = cosine); add / remove document |
| `retrieval.py` | Question → top-k chunks with rank and score |
| `prompts.py` | System rules, source format, injection patterns |
| `llm.py` | `LLMProvider` protocol; `AnthropicProvider`, `ExtractiveProvider` |
| `generation.py` | Evidence selection, prompt building, citation verification |
| `pipeline.py` | Facade used by the UI and the evaluation |
| `i18n.py`, `i18n_content.py` | Italian / English interface strings (a test checks both languages have the same keys) |
| `preview.py` | Renders a cited PDF page with the used sentences highlighted |
| `evaluation.py` | Repeatable evaluation with four separate metrics |

## Design decisions

- **No RAG framework.** Each step is ~30–60 lines you can read and test. Abstractions can come later.
- **Citations are verified, not trusted.** The model sees sources labelled `S1…Sn` and cites those ids. The code maps each id back to the real document/page and reports ids that do not exist; an answer with no valid citation is flagged as unverifiable.
- **Chunks never cross pages**, so a citation is always exact.
- **Exact search (`IndexFlatIP`).** For a few thousand chunks, approximate indexes add complexity and no benefit.
- **Keyword search is the default; neural search is optional.** The first public deployment on Streamlit Community Cloud crashed with "resource limits" while using a neural embedding model. Measured locally (Windows): the full-precision multilingual model costs ~560 MB and the process ~790 MB; an 8-bit quantised export of the same model brings the whole app to 636 MB (embeddings close to the original, mean cosine 0.994 on a small sample, but retrieval slightly worse). Streamlit's [documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) lists a **guaranteed minimum of 690 MB** per app (maximum 2.7 GB, subject to change), so even the quantised model sat on the edge. Keyword search (stemmed words hashed into a vector, no model) uses ~225 MB for the whole app. The neural mode stays available for local use; `FastEmbedEmbedder(quantized=False)` gives the original model.
- **Provider interface** (`complete(system, user) -> str`): swapping vendor means writing one small class.
- **Offline extractive engine.** Deterministic, always grounded (it only copies sentences from the sources), keyless: it makes the demo, the tests and the evaluation reproducible, and gives a baseline to compare a real LLM against.
- **Prompt-injection handling is layered**: (1) suspicious sentences are removed before prompting and the user gets a warning; (2) the system prompt declares source text untrusted; (3) sources are escaped so a document cannot close the `<source>` tag; (4) citations are verified afterwards. This reduces the risk; it does not eliminate it.

## Evaluation

43 questions over 4 synthetic documents (`data/sample_pdfs/`, generated by `scripts/make_sample_pdfs.py`): 35 answerable and 8 not answerable. Categories: factual, procedure, Italian, safety, conflicting documents, prompt injection, unanswerable. Each question lists expected keywords and the gold document/page (`eval/questions.jsonl`).

Four metrics are reported separately, for both search methods. Configuration: chunk 400 / overlap 80, top-k 5, offline extractive engine; minimum similarity 0.10 (keyword) and 0.25 (semantic).

| Metric | Keyword (default) | Semantic |
|---|---|---|
| Retrieval hit@1 (a gold page is ranked first) | 0.97 | 0.77 |
| Retrieval hit@3 / hit@5 | 1.00 / 1.00 | 0.94 / 1.00 |
| Retrieval MRR | 0.98 | 0.87 |
| Answer correct (expected keywords present, no forbidden content), answerable questions | 0.89 (31/35) | 0.86 (30/35) |
| Citation hit (a cited page is a gold page), answered questions | 1.00 | 1.00 |
| Citation precision (cited pages that are gold) | 0.99 | 0.99 |
| Correct abstention on unanswerable questions | 0.88 (7/8) | 0.88 (7/8) |
| False abstention on answerable questions | 0.09 (3/35) | 0.11 (4/35) |

**The trade-off is real.** Raising the extractive engine's overlap threshold makes it refuse more unanswerable questions but also more answerable ones (columns: answer correct / correct abstention / false abstention):

| min overlap | Keyword | Semantic |
|---|---|---|
| 0.40 | 0.91 / 0.50 / 0.06 | 0.89 / 0.38 / 0.09 |
| 0.45 | 0.91 / 0.50 / 0.06 | 0.89 / 0.50 / 0.09 |
| **0.50 (default)** | 0.89 / 0.88 / 0.09 | 0.86 / 0.88 / 0.11 |
| 0.55 | 0.74 / 0.88 / 0.23 | 0.69 / 0.88 / 0.29 |
| 0.60 | 0.69 / 1.00 / 0.29 | 0.63 / 1.00 / 0.34 |

For maintenance documentation a wrong answer is worse than a refusal, which is why the default sits where refusals jump (0.50).

**What the evaluation showed**

- **The keyword search beats the neural one on this benchmark, and that proves less than it seems.** I wrote the questions myself, reusing words from the manuals, which favours word matching. A neural model should do better when the question paraphrases the manual; that is unmeasured here. Real manuals and real questions are needed to compare the two fairly.
- Similarity alone does **not** separate "no answer" from "answer": unanswerable questions on a covered topic score as high as answerable ones. The similarity threshold only catches off-topic questions. Refusing on-topic questions requires the answering engine to notice that the specific fact is missing.
- Observed errors in the default (keyword) run:
  - 3 false abstentions (q07 "which grease", q23 "che olio si usa", q27 "cosa fare quando compare l'allarme E02"): the question shares too few words with the sentence that answers it, and the offline engine cannot paraphrase.
  - 1 wrong answer on an answerable question (q21, "portata del compressore C-50"): it quoted a different sentence than the one holding the value.
  - 1 unanswerable question was answered (q37, "peso del compressore C-50"): it quoted a sentence that shares the product name but not the asked fact.
- The conflict and injection questions behave as intended: both conflicting values are reported with their citations, and the injected "100 bar" never reaches the answer.
- A bug found by the evaluation and fixed: the first version dropped a whole passage when it contained one injected sentence, which also discarded legitimate facts next to it (fire assembly point, emergency number). Only the suspicious sentences are removed now.
- A weakness found by looking at errors: the offline engine missed word forms ("lubricated" vs "Lubricate"). Sharing one crude stemmer between search and engine, and treating "often" as a question word, raised answer correctness. Those two changes were made after seeing these very errors, so the benchmark is no longer an independent test of them.

## Limitations (please read)

- **Small, synthetic, self-written benchmark.** 43 questions over 11 chunks. hit@5 = 1.00 is nearly trivial when the corpus has 11 chunks; hit@1 is the more informative number. These results do **not** show the system is reliable on real manuals.
- **Thresholds and word lists were tuned on the same questions used to report results**, so the numbers are optimistic. A proper evaluation needs a separate test set from real documents.
- **Keyword search does not understand meaning:** it fails on synonyms and paraphrases and cannot match a question in one language with a document in another. The neural mode handles those in principle, at a memory cost that did not fit a free public server.
- **The Claude engine was not evaluated** against the live API here. Expected to be better at paraphrase and refusals, but that is unmeasured.
- The extractive engine cannot paraphrase, translate or combine facts.
- **No OCR.** Scanned PDFs are rejected with a clear message. Tables and multi-column layouts are extracted as plain text and may lose structure.
- No persistence: the index lives in the session and is rebuilt on restart. Exact search will not scale to very large collections.
- Prompt-injection defences are heuristic (pattern matching + prompt design) and can be bypassed.
- Memory figures were measured on Windows; Linux servers may differ.
- Not for safety-critical decisions: always verify against the cited page.

## Privacy and data handling

What the app does with an uploaded PDF:

- It is held **in the server's memory** for the session. The code never writes uploaded files, extracted text or questions to disk or to a database, and it does not log them.
- It is discarded when the user presses **Delete all documents now**, and automatically after **15 minutes without interaction** while the page is open (`DOC_IDLE_MINUTES`, `0` disables it). When the page is closed, Streamlit discards the session; this project cannot make that instantaneous, and there is no documented time after which the memory is released (a forum report mentions memory from closed sessions not being freed immediately), so the documents are not guaranteed to disappear at the very moment of closing.
- Upload is blocked until the user ticks a confirmation that the documents are not confidential and may be used for the test.
- With the **Claude** engine, the retrieved passages (not the whole file) are sent to Anthropic's API to write the answer. The app says so when that engine is selected. The offline engine sends nothing anywhere.

What this does **not** guarantee:

- On a public demo the app runs on a **third-party hosting platform** that this project does not control; its own logging and retention are outside the app's reach and were not verified here.
- A disclaimer informs users, it does not make an upload safe, and it is not legal advice. For confidential manuals, run the app on your own machine or server (keyword search and the offline engine need no internet) instead of using a public demo.
- **A public Community Cloud app cannot be hidden from search engines.** Streamlit's [documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/indexability) says public apps are indexed automatically (private apps are not), and there is no setting to opt out. Not publishing the URL reduces discovery, but it is not access control. Two ways to restrict access: make the app **private** in the Streamlit sharing settings (viewers sign in with an invited email), or set an **`APP_PASSWORD`** (environment variable or Streamlit secret): the app then shows only a password box until it is typed. The password is a single shared secret without rate limiting, so it keeps casual visitors out; it is not strong authentication and does not change where the data is processed.
- A file being findable on a search engine does not mean it is free to redistribute: copyright still applies. For that reason no third-party manual is bundled in this repository.

## Deploying a public demo on Streamlit Community Cloud

1. Push the repository to GitHub (check that `.env` is not tracked).
2. On [share.streamlit.io](https://share.streamlit.io) create an app from the repo: main file `app/streamlit_app.py`; dependencies are read from `requirements.txt`.
3. Keep the default **keyword search**: the semantic mode is likely to exceed the free memory limit (see Design decisions).

A public demo is visible to anyone with the URL: use only non-confidential documents. Users can paste their own API key in the sidebar; do not put your key in the app's public configuration.

## Project structure

```
app/streamlit_app.py      Streamlit UI
src/rag_assistant/        library (see table above)
tests/                    pytest suite (offline)
eval/questions.jsonl      evaluation set
eval/results_*.json       latest evaluation results, one file per search method
data/sample_pdfs/         synthetic manuals (tests and evaluation; upload them to try the app)
scripts/                  sample PDF generator
docs/STUDY_GUIDE.md       guide to every module (Italian)
```

## License

MIT, see [LICENSE](LICENSE). The sample documents are invented for this project.
