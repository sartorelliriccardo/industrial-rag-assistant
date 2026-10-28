"""Long UI texts (About and Evaluation tabs, examples, preview). Merged into i18n.STRINGS.

Avoid curly braces in the markdown texts: they are format fields for `t(...)`.
"""

# Example questions stay in the language of the sample manuals, whatever the UI language:
# the offline engine cannot translate between the question and the document.
EXAMPLE_QUESTIONS = {
    "ex_normal": "What is the tightening torque of the impeller nut?",
    "ex_italian": "Qual e la pressione massima di esercizio del compressore C-50?",
    "ex_conflict": "What is the maximum operating pressure of the P-200 pump?",
    "ex_injection": "What is the internal emergency number?",
    "ex_noanswer": "What is the warranty period of the P-200?",
}

CONTENT: dict[str, dict[str, str]] = {
    "en": {
        "tab_assistant": "Assistant",
        "tab_about": "How it works",
        "tab_eval": "Evaluation",
        "examples_title": "Try an example (uses the sample manuals)",
        "ex_normal": "Normal question",
        "ex_italian": "Italian manual",
        "ex_conflict": "Documents disagree",
        "ex_injection": "Hidden instruction in a PDF",
        "ex_noanswer": "Answer not in the documents",
        "verify_values": "This answer contains technical values (torque, pressure, temperature...). Verify them on the cited page before acting.",
        "preview_title": "Page preview",
        "preview_caption": "{doc}, page {page}: the passage used is highlighted",
        "preview_no_highlight": "{doc}, page {page}: the passage could not be located on the page image",
        "preview_error": "Could not render the preview: {error}",
        "about_md": """
### What it is
Industrial RAG Assistant answers questions about technical manuals (PDF) and always shows **which document and page** each answer comes from. If the documents do not contain the answer, it says so instead of guessing.

### How to use it
1. Upload one or more PDFs in the sidebar, or load the sample manuals.
2. Ask a question in Italian or English.
3. Read the answer, then check the sources, the page preview and the retrieved passages.

### How it works
- **Reading:** the text of every page is extracted, keeping document name and page number.
- **Chunks:** pages are split into small overlapping pieces that never cross a page, so citations are exact.
- **Embeddings:** each piece becomes a vector that represents its meaning, also across Italian and English.
- **Retrieval:** the question becomes a vector and the most similar pieces are found.
- **Constrained answer:** the pieces are given to the answer engine with strict rules: answer only from them, cite the sources, say when the evidence is not enough.
- **Verification:** the code checks every citation against the passages that were really provided.

### Answer engines
- **Offline (default):** quotes the sentences that best match the question. No API key, always grounded in the text, but it cannot rephrase and fails when the question uses different words than the manual.
- **Claude (optional):** rewrites the answer from the passages, using your own API key (kept in memory only). It is implemented but **not yet evaluated** in this project.

### Safety measures
- Instructions hidden in a PDF (prompt injection) are removed from the evidence and flagged.
- If documents disagree, both values are reported with their sources.
- Answers with technical values remind you to verify them on the cited page.

### Limits
- The evaluation is small and synthetic: it does not prove reliability on real manuals.
- No OCR: scanned PDFs are rejected. Tables lose their structure.
- Do not use it for safety-critical decisions without checking the cited page.

### Privacy and code
PDFs are processed in memory and never stored. Do not upload confidential documents to a public demo. Source code: github.com/sartorelliriccardo/industrial-rag-assistant
""",
        "tab_arch": "Architecture",
        "arch_title": "Architecture: how it works",
        "arch_intro": "Two flows share the same index: ingestion runs once per PDF, the question flow runs every time you ask. Under each step you can see the code module that implements it.",
        "cluster_ingest": "Ingestion (once per PDF)",
        "cluster_query": "Every question",
        "flow_upload": "PDF upload",
        "flow_pages": "Pages with page number",
        "flow_chunks": "Overlapping chunks",
        "flow_vectors": "Embeddings (vectors)",
        "flow_index": "Vector index (cosine)",
        "flow_question": "Question",
        "flow_embed_q": "Embed the question",
        "flow_search": "Most similar passages (top-k)",
        "flow_filter": "Score threshold + injection filter",
        "flow_prompt": "Prompt: rules + untrusted sources",
        "flow_engine": "Answer engine (offline or Claude)",
        "flow_verify": "Citation check",
        "flow_answer": "Answer, sources, page preview",
        "flow_no": "Not enough information",
        "arch_md": """
### Components

| Step | Module | Technology | What it does |
|---|---|---|---|
| Reading | `pdf_loader.py` | PyMuPDF | Extracts the text of each page keeping document name and page number; clear errors for empty, corrupted, protected or scanned PDFs |
| Chunking | `chunking.py` | Python | Sliding window of 400 characters with 80 of overlap, cut on whitespace, never across a page |
| Embeddings | `embeddings.py` | FastEmbed (ONNX), multilingual MiniLM quantised to 8 bit, 384 dimensions | Turns text into a vector of unit length that represents its meaning, in Italian and English |
| Index | `index.py` | FAISS (exact inner-product index) | Stores the vectors; with unit vectors the inner product is the cosine similarity |
| Retrieval | `retrieval.py` | - | Finds the top-k most similar passages with their score |
| Generation | `generation.py`, `prompts.py` | - | Applies the score threshold, removes injected instructions, builds the constrained prompt and verifies the citations |
| Answer engine | `llm.py` | Offline extractive engine or Anthropic API | Writes the answer from the evidence only; swappable through a one-method interface |
| Evaluation | `evaluation.py` | - | Four separate metrics on 43 questions: retrieval, correctness, citations, refusals |
| Interface | `app/streamlit_app.py` | Streamlit | Upload, answer, sources, page preview, evaluation view |

### Design decisions
- **No RAG framework.** Every step is short, readable and tested, so each choice can be explained and measured.
- **Citations are verified, not trusted.** The engine cites ids like S1; the code maps each id back to the real document and page and reports any id that does not exist.
- **Chunks never cross pages**, so a citation always points to the exact page.
- **Exact search.** For thousands of passages an approximate index adds complexity and no benefit.
- **ONNX embeddings instead of PyTorch**: much lighter to install and to deploy.
- **Swappable engine.** The offline engine makes demo, tests and evaluation reproducible without keys; Claude plugs in through the same interface.
- **Layered defence against prompt injection:** suspicious sentences are removed, the prompt declares sources untrusted, source text is escaped, and citations are verified afterwards. It reduces the risk, it does not remove it.
- **Refuse rather than guess:** a low similarity score means the engine is never called.
""",
        "eval_title": "How well does it work?",
        "eval_intro": "Results of the repeatable evaluation (eval/questions.jsonl): {n} questions, {n_ans} with an answer in the documents and {n_unans} without. Offline engine, chunk {size}/{overlap}, top-{k}, minimum similarity {min_score}.",
        "eval_missing": "No evaluation results found. Run: python -m rag_assistant.evaluation --out eval/results.json",
        "m_hit1": "Right page ranked first",
        "m_hit3": "Right page in top 3",
        "m_mrr": "MRR (retrieval)",
        "m_correct": "Correct answers",
        "m_cit_hit": "Citation points to the right page",
        "m_cit_prec": "Citation precision",
        "m_abst": "Correct refusals",
        "m_false_abst": "Wrong refusals",
        "eval_problems": "Observed errors",
        "eval_no_problems": "No errors in this run.",
        "col_id": "ID",
        "col_category": "Category",
        "col_question": "Question",
        "col_problem": "Problem",
        "prob_unanswerable_answered": "Answered a question that has no answer in the documents",
        "prob_retrieval_miss": "Retrieval miss: the right page is not in the results",
        "prob_false_abstention": "Refused a question that has an answer",
        "prob_missing_keyword": "The answer lacks the expected content",
        "prob_forbidden": "The answer contains forbidden content (injected instruction followed?)",
        "prob_no_gold_citation": "No citation points to the right page",
        "eval_limits_md": """
**Read these numbers with care.** They come from a small, self-written benchmark over synthetic documents (11 passages in total), and the thresholds were tuned on the same questions, so they are optimistic. They do **not** show that the system is reliable on real manuals. The Claude engine has not been evaluated. Details and the refusal trade-off are in the README.
""",
    },
    "it": {
        "tab_assistant": "Assistente",
        "tab_about": "Come funziona",
        "tab_eval": "Valutazione",
        "examples_title": "Prova un esempio (usa i manuali di esempio)",
        "ex_normal": "Domanda normale",
        "ex_italian": "Manuale in italiano",
        "ex_conflict": "I documenti sono in disaccordo",
        "ex_injection": "Istruzione nascosta in un PDF",
        "ex_noanswer": "Risposta non presente nei documenti",
        "verify_values": "Questa risposta contiene valori tecnici (coppie, pressioni, temperature...). Verificali sulla pagina citata prima di intervenire.",
        "preview_title": "Anteprima della pagina",
        "preview_caption": "{doc}, pagina {page}: il passaggio usato è evidenziato",
        "preview_no_highlight": "{doc}, pagina {page}: non sono riuscito a localizzare il passaggio sull'immagine della pagina",
        "preview_error": "Impossibile creare l'anteprima: {error}",
        "about_md": """
### Che cos'è
Industrial RAG Assistant risponde a domande sui manuali tecnici (PDF) e mostra sempre **da quale documento e pagina** viene ogni risposta. Se i documenti non contengono la risposta, lo dice invece di inventare.

### Come si usa
1. Carica uno o più PDF dalla barra laterale, oppure usa i manuali di esempio.
2. Fai una domanda in italiano o in inglese.
3. Leggi la risposta, poi controlla le fonti, l'anteprima della pagina e i passaggi recuperati.

### Come funziona
- **Lettura:** si estrae il testo di ogni pagina, tenendo nome del documento e numero di pagina.
- **Chunk:** le pagine sono divise in pezzi piccoli e sovrapposti che non attraversano mai una pagina, così le citazioni sono esatte.
- **Embedding:** ogni pezzo diventa un vettore che ne rappresenta il significato, anche tra italiano e inglese.
- **Ricerca:** la domanda diventa un vettore e si trovano i pezzi più simili.
- **Risposta vincolata:** i pezzi vengono dati al motore di risposta con regole precise: rispondere solo da lì, citare le fonti, dire quando le evidenze non bastano.
- **Verifica:** il codice controlla ogni citazione rispetto ai passaggi realmente forniti.

### Motori di risposta
- **Offline (predefinito):** cita le frasi che meglio corrispondono alla domanda. Nessuna chiave API, sempre ancorato al testo, ma non riformula e sbaglia quando la domanda usa parole diverse dal manuale.
- **Claude (opzionale):** riscrive la risposta a partire dai passaggi, con la tua chiave API (resta solo in memoria). È implementato ma **non ancora valutato** in questo progetto.

### Misure di sicurezza
- Le istruzioni nascoste in un PDF (prompt injection) vengono tolte dalle evidenze e segnalate.
- Se i documenti sono in disaccordo, si riportano entrambi i valori con le fonti.
- Le risposte con valori tecnici ricordano di verificarli sulla pagina citata.

### Limiti
- La valutazione è piccola e sintetica: non prova l'affidabilità su manuali reali.
- Niente OCR: i PDF scansionati vengono rifiutati. Le tabelle perdono la struttura.
- Non usarlo per decisioni di sicurezza senza controllare la pagina citata.

### Privacy e codice
I PDF sono elaborati in memoria e mai salvati. Non caricare documenti riservati in una demo pubblica. Codice sorgente: github.com/sartorelliriccardo/industrial-rag-assistant
""",
        "tab_arch": "Architettura",
        "arch_title": "Architettura: come funziona",
        "arch_intro": "Due flussi condividono lo stesso indice: il caricamento avviene una volta per PDF, il flusso della domanda a ogni richiesta. Sotto ogni passaggio vedi il modulo di codice che lo realizza.",
        "cluster_ingest": "Caricamento (una volta per PDF)",
        "cluster_query": "A ogni domanda",
        "flow_upload": "Caricamento PDF",
        "flow_pages": "Pagine con numero di pagina",
        "flow_chunks": "Chunk sovrapposti",
        "flow_vectors": "Embedding (vettori)",
        "flow_index": "Indice vettoriale (coseno)",
        "flow_question": "Domanda",
        "flow_embed_q": "Vettorizza la domanda",
        "flow_search": "Passaggi più simili (top-k)",
        "flow_filter": "Soglia di punteggio + filtro injection",
        "flow_prompt": "Prompt: regole + fonti non attendibili",
        "flow_engine": "Motore di risposta (offline o Claude)",
        "flow_verify": "Verifica delle citazioni",
        "flow_answer": "Risposta, fonti, anteprima pagina",
        "flow_no": "Informazioni insufficienti",
        "arch_md": """
### Componenti

| Passaggio | Modulo | Tecnologia | Cosa fa |
|---|---|---|---|
| Lettura | `pdf_loader.py` | PyMuPDF | Estrae il testo di ogni pagina tenendo nome del documento e numero di pagina; errori chiari per PDF vuoti, danneggiati, protetti o scansionati |
| Chunking | `chunking.py` | Python | Finestra scorrevole di 400 caratteri con 80 di sovrapposizione, tagliata sugli spazi, mai a cavallo di una pagina |
| Embedding | `embeddings.py` | FastEmbed (ONNX), MiniLM multilingue quantizzato a 8 bit, 384 dimensioni | Trasforma il testo in un vettore di lunghezza unitaria che ne rappresenta il significato, in italiano e inglese |
| Indice | `index.py` | FAISS (indice esatto a prodotto scalare) | Conserva i vettori; con vettori unitari il prodotto scalare è la similarità coseno |
| Ricerca | `retrieval.py` | - | Trova i top-k passaggi più simili con il loro punteggio |
| Generazione | `generation.py`, `prompts.py` | - | Applica la soglia di punteggio, toglie le istruzioni iniettate, costruisce il prompt vincolato e verifica le citazioni |
| Motore di risposta | `llm.py` | Motore offline estrattivo o API Anthropic | Scrive la risposta solo dalle evidenze; sostituibile tramite un'interfaccia a un solo metodo |
| Valutazione | `evaluation.py` | - | Quattro metriche separate su 43 domande: ricerca, correttezza, citazioni, rifiuti |
| Interfaccia | `app/streamlit_app.py` | Streamlit | Caricamento, risposta, fonti, anteprima della pagina, vista della valutazione |

### Scelte di progetto
- **Nessun framework RAG.** Ogni passaggio è breve, leggibile e testato, così ogni scelta si può spiegare e misurare.
- **Le citazioni si verificano, non ci si fida.** Il motore cita id come S1; il codice riporta ogni id al documento e alla pagina reali e segnala quelli che non esistono.
- **I chunk non attraversano mai le pagine**, quindi una citazione punta sempre alla pagina esatta.
- **Ricerca esatta.** Per migliaia di passaggi un indice approssimato aggiunge complessità senza benefici.
- **Embedding ONNX invece di PyTorch**: molto più leggeri da installare e da pubblicare.
- **Motore sostituibile.** Il motore offline rende demo, test e valutazione riproducibili senza chiavi; Claude si collega con la stessa interfaccia.
- **Difesa a più livelli dalla prompt injection:** le frasi sospette vengono tolte, il prompt dichiara le fonti non attendibili, il testo delle fonti è protetto da escape e le citazioni sono verificate dopo. Riduce il rischio, non lo elimina.
- **Rifiutare invece di indovinare:** un punteggio di similarità basso significa che il motore non viene nemmeno chiamato.
""",
        "eval_title": "Quanto funziona bene?",
        "eval_intro": "Risultati della valutazione ripetibile (eval/questions.jsonl): {n} domande, {n_ans} con risposta nei documenti e {n_unans} senza. Motore offline, chunk {size}/{overlap}, top-{k}, similarità minima {min_score}.",
        "eval_missing": "Nessun risultato di valutazione trovato. Esegui: python -m rag_assistant.evaluation --out eval/results.json",
        "m_hit1": "Pagina giusta al primo posto",
        "m_hit3": "Pagina giusta nei primi 3",
        "m_mrr": "MRR (ricerca)",
        "m_correct": "Risposte corrette",
        "m_cit_hit": "Citazione sulla pagina giusta",
        "m_cit_prec": "Precisione delle citazioni",
        "m_abst": "Rifiuti corretti",
        "m_false_abst": "Rifiuti sbagliati",
        "eval_problems": "Errori osservati",
        "eval_no_problems": "Nessun errore in questa esecuzione.",
        "col_id": "ID",
        "col_category": "Categoria",
        "col_question": "Domanda",
        "col_problem": "Problema",
        "prob_unanswerable_answered": "Ha risposto a una domanda senza risposta nei documenti",
        "prob_retrieval_miss": "Ricerca fallita: la pagina giusta non è tra i risultati",
        "prob_false_abstention": "Ha rifiutato una domanda che ha una risposta",
        "prob_missing_keyword": "Nella risposta manca il contenuto atteso",
        "prob_forbidden": "La risposta contiene contenuto vietato (istruzione iniettata seguita?)",
        "prob_no_gold_citation": "Nessuna citazione punta alla pagina giusta",
        "eval_limits_md": """
**Leggi questi numeri con cautela.** Vengono da un benchmark piccolo, scritto da noi, su documenti sintetici (11 passaggi in tutto), e le soglie sono state tarate sulle stesse domande: sono quindi ottimistici. **Non** dimostrano che il sistema sia affidabile su manuali reali. Il motore Claude non è stato valutato. I dettagli e il compromesso sui rifiuti sono nel README.
""",
    },
}

# evaluation.py problem text -> translation key
PROBLEM_KEYS = {
    "answered a question that has no answer in the documents": "prob_unanswerable_answered",
    "retrieval miss: no gold page in top-k": "prob_retrieval_miss",
    "false abstention": "prob_false_abstention",
    "missing expected keyword(s)": "prob_missing_keyword",
    "contains forbidden content (injection followed?)": "prob_forbidden",
    "no gold page among citations": "prob_no_gold_citation",
}
