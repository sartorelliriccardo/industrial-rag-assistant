"""UI strings in Italian and English. Add a language by adding a dict with the same keys."""

from rag_assistant.i18n_content import CONTENT, PROBLEM_KEYS  # noqa: F401
from rag_assistant.i18n_walk import WALK

LANGUAGES = {"it": "Italiano", "en": "English"}
DEFAULT_LANGUAGE = "it"

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "title": "Industrial RAG Assistant",
        "subtitle": (
            "Ask questions about your technical manuals. Every answer cites document and page; "
            "if the documents do not contain the answer, the assistant says so."
        ),
        "language": "Language",
        "settings": "Settings",
        "engine": "Answer engine",
        "engine_extractive": "Offline (quotes the documents, no API key)",
        "engine_anthropic": "Claude (API key required)",
        "api_key": "Anthropic API key",
        "api_key_help": "Kept in memory only.",
        "need_key": "Enter an API key in the sidebar, or switch to the offline answer engine.",
        "params": "Retrieval parameters",
        "chunk_size": "Chunk size (chars)",
        "chunk_overlap": "Chunk overlap (chars)",
        "top_k": "Passages retrieved (top-k)",
        "min_score": "Minimum similarity",
        "params_note": "Changing chunk size/overlap applies to documents added afterwards.",
        "documents": "Documents",
        "upload": "Upload PDF manuals",
        "privacy": (
            "Your files are processed in the server's memory for this session only. "
            "This app does not write them to disk or to any database, and they are deleted "
            "a few seconds after you close the page, or when you press the delete button. "
            "The app runs on a third-party hosting platform that we do not control. "
            "This is a public demo: do not upload confidential documents."
        ),
        "confirm_rights": "I confirm that these documents are not confidential and that I may use them for this test.",
        "clear_all": "Delete all documents now",
        "cleared": "Documents deleted from memory.",
        "idle_note": "Documents are also deleted automatically after {minutes} minutes of inactivity, as long as this page stays open.",
        "idle_cleared": "Documents deleted after {minutes} minutes of inactivity.",
        "claude_privacy": "With Claude, the passages found in your documents are sent to Anthropic's API to write the answer. Use the offline engine for anything sensitive.",
        "indexing": "Indexing {name}...",
        "indexed": "{name}: {n} passages indexed",
        "indexed_skipped": "{name}: {n} passages indexed (index page(s) left out: {pages})",
        "err_corrupted": "{name}: the file is empty, corrupted or password protected.",
        "err_no_text": "{name}: no extractable text found (scanned PDF? OCR is not supported).",
        "err_not_found": "{name}: file not found.",
        "err_pdf": "{name}: could not read this PDF.",
        "start_hint": "Upload one or more PDF manuals in the sidebar to start.",
        "indexed_docs": "Indexed documents ({n}) - {chunks} passages",
        "remove": "Remove",
        "question": "Your question",
        "placeholder": "e.g. What is the tightening torque of the impeller nut?",
        "searching": "Searching the documents...",
        "answer_error": "Could not get an answer: {error}",
        "insufficient": (
            "The uploaded documents do not contain enough information to answer this question."
        ),
        "uncited": "The model answered without valid citations, so the answer cannot be verified:",
        "sources": "Sources",
        "source_line": "**{doc}**, page {page}",
        "invalid_citations": "The model cited non-existent sources: {ids}",
        "flagged": (
            "{n} passage(s) contained sentences that look like instructions aimed at the AI "
            "(possible prompt injection); those sentences were excluded from the answer."
        ),
        "passages": "Retrieved passages",
        "passages_help": "Check the evidence yourself. Score = cosine similarity (higher is closer).",
        "tag_used": "sent to answer engine",
        "tag_unused": "not sent (low score/flagged)",
        "passage_title": "#{rank} · {doc}, p. {page} · score {score:.2f} · {tag}",
    },
    "it": {
        "title": "Assistente RAG per manualistica industriale",
        "subtitle": (
            "Fai domande sui tuoi manuali tecnici. Ogni risposta cita documento e pagina; "
            "se i documenti non contengono la risposta, l'assistente lo dice."
        ),
        "language": "Lingua",
        "settings": "Impostazioni",
        "engine": "Motore di risposta",
        "engine_extractive": "Offline (cita i documenti, nessuna chiave API)",
        "engine_anthropic": "Claude (serve una chiave API)",
        "api_key": "Chiave API Anthropic",
        "api_key_help": "Resta solo in memoria.",
        "need_key": "Inserisci una chiave API nella barra laterale, oppure scegli il motore offline.",
        "params": "Parametri di ricerca",
        "chunk_size": "Dimensione dei chunk (caratteri)",
        "chunk_overlap": "Sovrapposizione dei chunk (caratteri)",
        "top_k": "Passaggi recuperati (top-k)",
        "min_score": "Similarità minima",
        "params_note": (
            "Dimensione e sovrapposizione valgono per i documenti aggiunti dopo la modifica."
        ),
        "documents": "Documenti",
        "upload": "Carica i manuali in PDF",
        "privacy": (
            "I tuoi file sono elaborati nella memoria del server solo per questa sessione. "
            "L'app non li scrive su disco né in alcun database, e vengono eliminati pochi "
            "secondi dopo la chiusura della pagina, o quando premi il pulsante di cancellazione. "
            "L'app gira su una piattaforma di hosting di terzi che non controlliamo. "
            "Questa è una demo pubblica: non caricare documenti riservati."
        ),
        "confirm_rights": "Confermo che i documenti non sono riservati e che posso usarli per questa prova.",
        "clear_all": "Cancella ora tutti i documenti",
        "cleared": "Documenti cancellati dalla memoria.",
        "idle_note": "I documenti vengono anche cancellati automaticamente dopo {minutes} minuti di inattività, finché questa pagina resta aperta.",
        "idle_cleared": "Documenti cancellati dopo {minutes} minuti di inattività.",
        "claude_privacy": "Con Claude, i passaggi trovati nei tuoi documenti vengono inviati all'API di Anthropic per scrivere la risposta. Per qualsiasi cosa sensibile usa il motore offline.",
        "indexing": "Indicizzazione di {name}...",
        "indexed": "{name}: {n} passaggi indicizzati",
        "indexed_skipped": "{name}: {n} passaggi indicizzati (pagine d'indice escluse: {pages})",
        "err_corrupted": "{name}: il file è vuoto, danneggiato o protetto da password.",
        "err_no_text": "{name}: nessun testo estraibile (PDF scansionato? L'OCR non è supportato).",
        "err_not_found": "{name}: file non trovato.",
        "err_pdf": "{name}: impossibile leggere questo PDF.",
        "start_hint": "Carica uno o più manuali in PDF dalla barra laterale per iniziare.",
        "indexed_docs": "Documenti indicizzati ({n}) - {chunks} passaggi",
        "remove": "Rimuovi",
        "question": "La tua domanda",
        "placeholder": "es. Qual è la coppia di serraggio del dado della girante?",
        "searching": "Ricerca nei documenti...",
        "answer_error": "Impossibile ottenere una risposta: {error}",
        "insufficient": (
            "I documenti caricati non contengono informazioni sufficienti per rispondere "
            "a questa domanda."
        ),
        "uncited": "Il modello ha risposto senza citazioni valide, quindi la risposta non è verificabile:",
        "sources": "Fonti",
        "source_line": "**{doc}**, pagina {page}",
        "invalid_citations": "Il modello ha citato fonti inesistenti: {ids}",
        "flagged": (
            "{n} passaggio/i contenevano frasi che sembrano istruzioni rivolte all'IA "
            "(possibile prompt injection): quelle frasi sono state escluse dalla risposta."
        ),
        "passages": "Passaggi recuperati",
        "passages_help": (
            "Controlla tu le evidenze. Punteggio = similarità coseno (più alto = più vicino)."
        ),
        "tag_used": "inviato al motore di risposta",
        "tag_unused": "non inviato (punteggio basso/segnalato)",
        "passage_title": "#{rank} · {doc}, p. {page} · punteggio {score:.2f} · {tag}",
    },
}


for _lang, _extra in CONTENT.items():
    STRINGS[_lang].update(_extra)
for _lang, _extra in WALK.items():
    STRINGS[_lang].update(_extra)


def t(lang: str, key: str, **kwargs) -> str:
    """Translate `key` into `lang`, falling back to English if the language is unknown."""
    table = STRINGS.get(lang, STRINGS["en"])
    return table[key].format(**kwargs) if kwargs else table[key]
