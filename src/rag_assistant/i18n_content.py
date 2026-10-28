"""Long UI texts (About and Evaluation tabs, examples, preview). Merged into i18n.STRINGS.

Avoid curly braces in the markdown texts: they are format fields for `t(...)`.
"""

CONTENT: dict[str, dict[str, str]] = {
    "en": {
        "tab_assistant": "Assistant",
        "tab_about": "How it works",
        "tab_eval": "Evaluation",
        "search": "Search method",
        "search_keyword": "By keywords (light, default)",
        "search_semantic": "By meaning (neural model)",
        "search_semantic_note": "Loads a ~120 MB model and needs about 400 MB more memory: it may not fit on a free public server. Best used locally.",
        "verify_values": "This answer contains technical values (torque, pressure, temperature...). Verify them on the cited page before acting.",
        "preview_title": "Page preview",
        "preview_caption": "{doc}, page {page}: the passage used is highlighted",
        "preview_no_highlight": "{doc}, page {page}: the passage could not be located on the page image",
        "preview_error": "Could not render the preview: {error}",
        "about_md": """
### What it is
A tool that answers questions about your technical manuals (PDF) and **always shows where the answer comes from**: document and page. If the answer is not in the documents, it says so instead of making something up.

### How to use it
1. **Add documents.** Upload your PDF manuals in the left sidebar.
2. **Ask a question** in Italian or English.
3. **Check the answer.** Under it you see the source, the PDF page with the sentence highlighted, and the passages the assistant found.

### What happens when you ask
1. **It searches** the documents for the passages closest to your question.
2. **It writes the answer** using only those passages.
3. **It checks itself:** every source it names must be a passage it was really given.
4. **It says "not enough information"** when nothing relevant is found.

*Example:* you ask "What is the tightening torque of the impeller nut?" → it finds page 3 of the pump manual → it answers "120 N.m" → it shows you page 3 with that sentence highlighted.

### Two ways to answer
- **Offline (default):** copies the best-matching sentences from the manual. Free, no account needed.
- **Claude (optional):** writes a more natural answer, using your own API key. Not tested yet in this project.

### What it does not do
- It cannot read scanned PDFs (no OCR).
- It does not understand tables well.
- It does not replace checking the page: always verify torques, pressures and temperatures on the cited page.

### Privacy
Your PDFs are used in memory only and are never saved. Do not upload confidential documents to a public demo.

Technical details are in the **Architecture** tab, and how well it works is in the **Evaluation** tab.
""",
        "tab_arch": "Architecture",
        "arch_title": "Architecture: how it works",
        "arch_intro": "Two flows share the same index: ingestion runs once per PDF, the question flow runs every time you ask. Under each step you can see the code module that implements it.",
        "cluster_ingest": "Ingestion (once per PDF)",
        "cluster_query": "Every question",
        "flow_upload": "PDF upload",
        "flow_pages": "Pages + page numbers",
        "flow_chunks": "Overlapping chunks",
        "flow_vectors": "Embeddings (vectors)",
        "flow_index": "Vector index (cosine)",
        "flow_question": "Question",
        "flow_embed_q": "Embed the question",
        "flow_search": "Top-k similar passages",
        "flow_filter": "Threshold + injection filter",
        "flow_prompt": "Prompt: rules + sources",
        "flow_engine": "Offline or Claude engine",
        "flow_verify": "Citation check",
        "flow_answer": "Answer + sources",
        "flow_no": "Not enough information",
        "arch_md": """
### Components

| Step | Module | Technology | What it does |
|---|---|---|---|
| Reading | `pdf_loader.py` | PyMuPDF | Extracts the text of each page keeping document name and page number; clear errors for empty, corrupted, protected or scanned PDFs |
| Chunking | `chunking.py` | Python | Sliding window of 400 characters with 80 of overlap, cut on whitespace, never across a page |
| Embeddings | `embeddings.py` | Default: stemmed keywords hashed into a vector (no model). Optional: FastEmbed (ONNX), multilingual MiniLM quantised to 8 bit | Turns text into a unit-length vector. Keyword mode matches shared words; the neural mode matches meaning, also across Italian and English |
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
        "eval_intro": "Results of the repeatable evaluation (eval/questions.jsonl): {n} questions, {n_ans} with an answer in the documents and {n_unans} without. Search: {mode}. Offline answer engine, chunk {size}/{overlap}, top-{k}, minimum similarity {min_score}.",
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
        "search": "Metodo di ricerca",
        "search_keyword": "Per parole chiave (leggero, predefinito)",
        "search_semantic": "Per significato (modello neurale)",
        "search_semantic_note": "Carica un modello da ~120 MB e richiede circa 400 MB di memoria in più: su un server pubblico gratuito potrebbe non starci. Meglio usarlo in locale.",
        "verify_values": "Questa risposta contiene valori tecnici (coppie, pressioni, temperature...). Verificali sulla pagina citata prima di intervenire.",
        "preview_title": "Anteprima della pagina",
        "preview_caption": "{doc}, pagina {page}: il passaggio usato è evidenziato",
        "preview_no_highlight": "{doc}, pagina {page}: non sono riuscito a localizzare il passaggio sull'immagine della pagina",
        "preview_error": "Impossibile creare l'anteprima: {error}",
        "about_md": """
### Che cos'è
Uno strumento che risponde a domande sui tuoi manuali tecnici (PDF) e **mostra sempre da dove viene la risposta**: documento e pagina. Se la risposta non è nei documenti, lo dice invece di inventare.

### Come si usa
1. **Aggiungi i documenti.** Carica i tuoi manuali in PDF dalla barra laterale a sinistra.
2. **Fai una domanda** in italiano o in inglese.
3. **Controlla la risposta.** Sotto trovi la fonte, la pagina del PDF con la frase evidenziata e i passaggi che l'assistente ha trovato.

### Cosa succede quando fai una domanda
1. **Cerca** nei documenti i passaggi più vicini alla tua domanda.
2. **Scrive la risposta** usando solo quei passaggi.
3. **Si controlla da solo:** ogni fonte che nomina deve essere un passaggio che ha davvero ricevuto.
4. **Dice «informazioni insufficienti»** quando non trova nulla di pertinente.

*Esempio:* chiedi «Qual è la coppia di serraggio del dado della girante?» → trova la pagina 3 del manuale della pompa → risponde «120 N.m» → ti mostra la pagina 3 con quella frase evidenziata.

### Due modi di rispondere
- **Offline (predefinito):** copia dal manuale le frasi che meglio corrispondono. Gratis, senza account.
- **Claude (opzionale):** scrive una risposta più naturale, con la tua chiave API. In questo progetto non è ancora stato provato.

### Cosa non fa
- Non legge i PDF scansionati (niente OCR).
- Non capisce bene le tabelle.
- Non sostituisce il controllo della pagina: verifica sempre coppie, pressioni e temperature sulla pagina citata.

### Privacy
I tuoi PDF restano solo in memoria e non vengono mai salvati. Non caricare documenti riservati in una demo pubblica.

I dettagli tecnici sono nella scheda **Architettura**, e quanto funziona bene nella scheda **Valutazione**.
""",
        "tab_arch": "Architettura",
        "arch_title": "Architettura: come funziona",
        "arch_intro": "Due flussi condividono lo stesso indice: il caricamento avviene una volta per PDF, il flusso della domanda a ogni richiesta. Sotto ogni passaggio vedi il modulo di codice che lo realizza.",
        "cluster_ingest": "Caricamento (una volta per PDF)",
        "cluster_query": "A ogni domanda",
        "flow_upload": "Caricamento PDF",
        "flow_pages": "Pagine + n. di pagina",
        "flow_chunks": "Chunk sovrapposti",
        "flow_vectors": "Embedding (vettori)",
        "flow_index": "Indice vettoriale (coseno)",
        "flow_question": "Domanda",
        "flow_embed_q": "Vettorizza la domanda",
        "flow_search": "Top-k passaggi simili",
        "flow_filter": "Soglia + filtro injection",
        "flow_prompt": "Prompt: regole + fonti",
        "flow_engine": "Motore offline o Claude",
        "flow_verify": "Verifica delle citazioni",
        "flow_answer": "Risposta + fonti",
        "flow_no": "Informazioni insufficienti",
        "arch_md": """
### Componenti

| Passaggio | Modulo | Tecnologia | Cosa fa |
|---|---|---|---|
| Lettura | `pdf_loader.py` | PyMuPDF | Estrae il testo di ogni pagina tenendo nome del documento e numero di pagina; errori chiari per PDF vuoti, danneggiati, protetti o scansionati |
| Chunking | `chunking.py` | Python | Finestra scorrevole di 400 caratteri con 80 di sovrapposizione, tagliata sugli spazi, mai a cavallo di una pagina |
| Embedding | `embeddings.py` | Predefinito: parole chiave radicalizzate in un vettore (nessun modello). Opzionale: FastEmbed (ONNX), MiniLM multilingue quantizzato a 8 bit | Trasforma il testo in un vettore di lunghezza unitaria. La modalità per parole chiave trova le parole in comune; quella neurale trova il significato, anche tra italiano e inglese |
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
        "eval_intro": "Risultati della valutazione ripetibile (eval/questions.jsonl): {n} domande, {n_ans} con risposta nei documenti e {n_unans} senza. Ricerca: {mode}. Motore di risposta offline, chunk {size}/{overlap}, top-{k}, similarità minima {min_score}.",
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
