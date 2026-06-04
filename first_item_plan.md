# Piano — Fase 1 Adaptive Analysis Foundation

## Obiettivo
Aggiungere a XPDSS un layer osservativo che analizzi i blocchi audio in tempo reale e pubblichi metriche percettive stabili, senza modificare ancora il suono.

## Principi
- Nessun impatto udibile nella pipeline.
- TDD stretto per il nuovo modulo.
- Integrazione minima e sicura nel worker di analisi già esistente.
- Niente dipendenze da misurazioni binaurali.

## Step

### Step 1 — Test RED per il nuovo motore analitico
Creare `tests/test_adaptive_renderer.py` con test su segnali sintetici che verifichino:
- segnale mono/centrale → correlazione alta, side ratio basso, dialogue score maggiore
- segnale stereo largo/arioso → side ratio alto, immersion score maggiore
- segnale compresso/clippato → compression score maggiore di un segnale più dinamico

### Step 2 — Implementazione minima del modulo
Creare `adaptive_renderer.py` con:
- dataclass snapshot delle metriche
- classe `AdaptiveFeatureEngine`
- estrazione feature dai primi due canali e, se presenti, dai surround
- smoothing per evitare jitter
- metodo per esportare uno snapshot serializzabile

### Step 3 — Verifica GREEN del modulo
Eseguire i test del nuovo file e poi l’intera suite disponibile.

### Step 4 — Integrazione observe-only in XPDSS
Modificare `audio_analyzer.py` per:
- istanziare il motore analitico
- abilitarlo/disabilitarlo con flag di config
- invocarlo dentro `_analysis_worker`
- pubblicare l’ultimo snapshot in memoria
- non cambiare alcun parametro DSP

### Step 5 — Verifica di integrazione
Verificare:
- test automatici verdi
- compilazione sintattica di `audio_analyzer.py` e `adaptive_renderer.py`
- assenza di modifiche alla pipeline audio effettiva

## Risultato atteso
Alla fine di questa fase XPDSS avrà il primo pezzo del nuovo cervello adattivo: sa leggere il contenuto, ma non lo altera ancora.
