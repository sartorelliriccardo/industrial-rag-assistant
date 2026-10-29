# Guida di studio: cosa fa ogni parte e come spiegarla

Obiettivo: poter raccontare il progetto a un colloquio senza guardare il codice.
Per ogni modulo: **cosa fa**, **perché è fatto così**, **domande da colloquio**.

Flusso in una riga: `PDF → pagine → chunk → vettori → indice FAISS → retrieval → prompt vincolato → risposta con citazioni → valutazione`.

---

## 1. `pdf_loader.py` — dal PDF alle pagine
**Cosa fa.** `load_pdf_bytes(data, nome)` apre il PDF con PyMuPDF e restituisce una lista di `Page(doc_name, page_number, text)`. Il numero di pagina parte da 1. Le pagine senza testo vengono saltate, ma la numerazione resta quella reale. Gli spazi e gli a-capo vengono compattati.

**Errori gestiti** (eccezioni dedicate, con messaggio chiaro): file inesistente, file vuoto o corrotto, PDF protetto da password, PDF senza testo estraibile (scansione).

**Perché.** Il numero di pagina va conservato subito: è la base di ogni citazione. Un errore chiaro vale più di un crash: l'utente capisce che il suo PDF è una scansione.

**Colloquio.**
- *Cosa succede con un manuale scansionato?* → viene rifiutato con un messaggio esplicito; servirebbe OCR (non implementato, è un limite dichiarato).
- *Perché eccezioni personalizzate?* → l'interfaccia può mostrare il messaggio giusto senza intercettare errori generici.

## 2. `chunking.py` — spezzare il testo
**Cosa fa.** `split_page(page, chunk_size, overlap)` fa scorrere una finestra di `chunk_size` caratteri sul testo della pagina. Ogni chunk successivo riparte `overlap` caratteri prima della fine del precedente. Taglia sugli spazi, mai a metà parola.

**Perché.**
- Un LLM e un embedding lavorano meglio su pezzi piccoli e focalizzati che su pagine intere.
- L'**overlap** serve perché un fatto a cavallo di due chunk resti intero in almeno uno.
- Un chunk **non attraversa mai le pagine**, così la citazione indica sempre la pagina esatta.
- `overlap < chunk_size` è obbligatorio: il passo è `chunk_size - overlap`, se fosse ≤ 0 il ciclo non avanzerebbe.

**Compromesso sulla dimensione.** Chunk piccoli: più precisi ma perdono il contesto. Chunk grandi: più contesto ma più rumore e il vettore diventa una "media" di troppi argomenti. Non si indovina: si misura (script di valutazione con `--chunk-size`).

**Colloquio.** *Come hai scelto la dimensione dei chunk?* → ho un default (800/120) scelto misurando su un manuale vero da 110 pagine: con pezzi da 400 caratteri le risposte utili erano 5 su 9, con 800 sono salite a 6. Sul mio corpus sintetico (una pagina = un pezzo) la differenza non si vede, e lo dichiaro.

## 3. `embeddings.py` — dal testo ai vettori
**Concetto.** Un embedding è un vettore di numeri (qui 384) che codifica il significato del testo: frasi con significato simile finiscono vicine nello spazio, anche in lingue diverse (la frase italiana sul bullone e quella inglese hanno similarità 0,49, la pizza 0,08).

**Similarità coseno.** Misura l'angolo tra due vettori, non la lunghezza. Con vettori normalizzati a lunghezza 1 il coseno coincide con il prodotto scalare: per questo il codice normalizza sempre (`_normalize`).

**Tre implementazioni dietro la stessa interfaccia (`Embedder`).**
- `LexicalEmbedder` (**predefinito**): ricerca per parole chiave. Toglie le parole vuote, riduce le parole alle prime 6 lettere (così «lubricated» e «lubricate» coincidono) e le "spalma" in un vettore con un hash. Nessun modello, circa 225 MB di memoria totale per l'app. Non capisce sinonimi né parafrasi.
- `FastEmbedEmbedder` (opzionale): modello vero multilingue, in formato ONNX e quantizzato a 8 bit (niente PyTorch). Capisce il significato, anche tra italiano e inglese, ma pesa circa 400 MB in più: su Streamlit Cloud gratuito (minimo garantito 690 MB) la prima versione è andata fuori limite. Per questo oggi non è il predefinito.
- `HashingEmbedder`: ogni parola viene mandata in uno di 512 "cassetti" con un hash; deterministico, senza rete, solo lessicale. Serve ai test: così girano ovunque senza scaricare 220 MB.

**Colloquio.**
- *Perché non sentence-transformers?* → richiede PyTorch (pesante). fastembed fa la stessa cosa con ONNX; l'interfaccia permette di cambiare.
- *Cosa non capisce l'HashingEmbedder?* → i sinonimi: è solo parole in comune.

## 4. `index.py` — l'indice FAISS
**Cosa fa.** `VectorIndex` tiene insieme i chunk e i loro vettori. `IndexFlatIP` fa una ricerca **esatta** per prodotto scalare (= coseno sui vettori normalizzati). `add` aggiunge, `remove_document` toglie un documento ricostruendo l'indice (un indice "flat" non sa cancellare righe), `search` restituisce i k chunk migliori con punteggio.

**Perché.** Per poche migliaia di chunk la ricerca esatta è istantanea; un indice approssimato (IVF, HNSW) aggiungerebbe complessità senza benefici.

**Colloquio.** *Quando passeresti a un indice approssimato o a un database vettoriale?* → con centinaia di migliaia di chunk, o quando serve persistenza e filtri sui metadati.

## 5. `retrieval.py` — trovare le evidenze
**Cosa fa.** `retrieve(domanda, index, embedder, k)`: trasforma la domanda in vettore e restituisce i k chunk più simili con rango e punteggio.

**Baseline.** Solo ricerca vettoriale, nessun re-ranking. Si misura prima, si migliora dopo (es. ricerca ibrida con BM25) solo se i numeri lo giustificano.

**Colloquio.** *Come sai che il retrieval funziona?* → lo misuro separatamente dalla risposta: hit@k (una pagina giusta è tra i risultati) e MRR (in che posizione compare la prima). Se il retrieval sbaglia, nessun LLM può rimediare.

## 6. `prompts.py`, `llm.py`, `generation.py` — risposta vincolata e citazioni
**Passi in `generate_answer`:**
1. Si **tolgono le frasi sospette** (prompt injection) dai passaggi, e si avvisa l'utente.
2. Si tengono solo i passaggi con punteggio ≥ soglia. Se non ne resta nessuno, si risponde "informazioni insufficienti" **senza nemmeno chiamare l'LLM**.
3. Si costruisce il prompt: regole di sistema da una parte, passaggi numerati `S1…Sn` dentro tag `<source>` dall'altra, dichiarati "dati non attendibili".
4. L'LLM risponde citando `[S1]`. Il codice **verifica** ogni id: se esiste lo traduce in "documento, p. N"; se non esiste lo segnala; una risposta senza citazioni valide è marcata "uncited".
5. Se il modello scrive `INSUFFICIENT_EVIDENCE`, la risposta è "informazioni insufficienti".

**Prompt injection.** Un PDF può contenere frasi come "ignora le istruzioni precedenti". Difese a più livelli: filtro delle frasi, regola nel prompt, escape dei caratteri `<` `>` (un documento non può chiudere il tag e inventare fonti), verifica delle citazioni. Non si elimina il rischio, lo si riduce: è nei limiti del README.

**Conflitti.** Il prompt chiede di riportare entrambi i valori con le citazioni; nei dati di esempio il datasheet Rev A dice 12 bar, il manuale Rev C 10 bar.

**Provider.** `LLMProvider` ha un solo metodo `complete(system, user)`. `AnthropicProvider` chiama Claude; `ExtractiveProvider` è offline: sceglie le frasi dei passaggi che coprono meglio le parole della domanda (con peso maggiore alle parole rare) e rifiuta se la copertura è bassa.

**Colloquio.**
- *Come eviti che l'LLM inventi?* → vincolo nel prompt + soglia sul punteggio + verifica delle citazioni + valutazione del rifiuto.
- *Perché non fidarti delle citazioni del modello?* → possono essere inventate: il codice le confronta con i passaggi realmente forniti.
- *Perché un provider "estrattivo"?* → riproducibilità: demo, test e valutazione senza chiave API, e una baseline con cui confrontare un LLM vero.

## 7. `pipeline.py` — il collante
`RagPipeline` espone due operazioni: `add_pdf_bytes(dati, nome)` (carica, taglia, calcola i vettori, aggiorna l'indice; ricaricare lo stesso nome sostituisce il documento; un PDF difettoso non tocca l'indice) e `ask(domanda)`. La usano sia l'app sia la valutazione, quindi misuriamo esattamente ciò che l'utente usa.

## 8. `evaluation.py` — misurare, non credere
**Quattro misure separate** sulle 43 domande di `eval/questions.jsonl`:
1. **Retrieval**: hit@1, hit@3, hit@k, MRR.
2. **Correttezza**: la risposta contiene le parole chiave attese (e non quelle vietate, per l'injection).
3. **Citazioni**: hit (cita una pagina giusta) e precisione (quante citate sono giuste).
4. **Rifiuto**: rifiuta le domande senza risposta (e quante domande valide rifiuta per errore).

**Cosa abbiamo imparato** (è la parte più utile da raccontare):
- La similarità vettoriale **non** distingue "nessuna risposta" da "risposta": una domanda sullo stesso argomento ma su un dato assente ottiene punteggi alti. La soglia sul punteggio ferma solo le domande fuori tema.
- C'è un compromesso tra rifiutare troppo e rifiutare troppo poco: la tabella nel README mostra la curva. In ambito manutenzione un rifiuto costa meno di una risposta sbagliata.
- Un bug trovato dalla valutazione: scartare un intero passaggio per una frase iniettata eliminava anche fatti validi. Corretto rimuovendo solo la frase sospetta.

**Onestà sui limiti.** 43 domande, 11 chunk, soglie tarate sulle stesse domande: i numeri non dimostrano affidabilità su manuali reali. Claude via API non è stato valutato.

**Colloquio.** *Quanto è affidabile?* → "Su un benchmark piccolo e sintetico: ricerca hit@1 97%, risposte corrette 89%, rifiuti corretti 88% (con la ricerca per parole chiave; le domande le ho scritte io riusando parole dei manuali, quindi il confronto con la ricerca semantica non è equo). Non lo considero provato su documenti reali; ecco i limiti e gli errori osservati."

## 9. `app/` — l'interfaccia
Il file `streamlit_app.py` (circa 50 righe) monta solo la pagina: legge le scelte della barra laterale, apre l'archivio della sessione e passa tutto alle cinque schede. Ogni parte sta in un file accanto, con un nome che dice cosa fa:

| File | Cosa contiene |
|---|---|
| `ui_sidebar.py` | lingua, metodo di ricerca, motore, parametri, conferma e caricamento dei PDF |
| `ui_assistant.py` | scheda *Assistente*: domanda, risposta, fonti, anteprima della pagina, passaggi |
| `ui_walkthrough.py` | scheda *Passo per passo* (le nove fasi) |
| `ui_architecture.py` | scheda *Architettura*: i due diagrammi |
| `ui_evaluation.py` | scheda *Valutazione*: legge `eval/results_*.json` |
| `ui_privacy.py` | password facoltativa, archivio per sessione, cancellazione dei file, cancellazione dopo inattività |
| `ui_state.py` | cosa ricevono le schede (`Ctx`) e la creazione della pipeline |

Da presentare: il nucleo è in `src/rag_assistant/` (lettura, pezzi, indice, ricerca, risposta); questi file sono solo l'interfaccia.

Barra laterale: lingua, metodo di ricerca, motore di risposta, parametri, conferma che i documenti non sono riservati, upload dei PDF, pulsante «Cancella ora». Streamlit rilancia lo script a ogni interazione, quindi i dati dell'utente non stanno nelle variabili dello script ma in un archivio per sessione (`workspace.py`), che li elimina pochi secondi dopo la chiusura della pagina. Il modello neurale (se scelto) sta in `st.cache_resource` e si carica una volta sola. Gli errori dei PDF e delle chiamate API sono mostrati senza far crollare la pagina.

**Le cinque schede, e come usarle per spiegare il progetto.**
- *Assistente*: la domanda e la risposta con fonti, anteprima della pagina e passaggi.
- *Come funziona*: la spiegazione in parole semplici, per chi non è tecnico.
- *Passo per passo*: la scheda da usare in un colloquio. Scrivi una domanda e per ognuna delle nove fasi vedi i dati veri: le parole tenute (stemming), i pezzi, dove cade ogni parola nel vettore, la classifica con la soglia, le frasi nascoste rimosse, il prompt esatto, l'uscita grezza del motore e il controllo delle citazioni. Con il PDF che contiene la frase «ignore all previous instructions» si vede il filtro in azione.
- *Architettura*: i due flussi, i moduli e le scelte.
- *Valutazione*: i numeri e gli errori, compresi i limiti.

**Colloquio.** *Mostrami come funziona.* → apri «Passo per passo» con una domanda e racconta le nove fasi seguendo lo schermo: è la dimostrazione che il sistema non è una scatola nera e che ogni numero può essere verificato.

---

## Domande trasversali
- **Perché senza LangChain?** Per capire e poter testare ogni componente; sono poche decine di righe ciascuno.
- **Cosa miglioreresti per primo?** Un set di valutazione su documenti reali e separato da quello di taratura; poi ricerca ibrida (BM25 + vettori), re-ranking e OCR.
- **Come gestisci chiavi e dati?** Chiavi solo in variabili d'ambiente o nella sessione; `.env` in `.gitignore`; i PDF restano in memoria e non vengono salvati.
- **Come testi senza API?** `ScriptedProvider` nei test, `HashingEmbedder`, PDF generati al volo: 303 test offline.
