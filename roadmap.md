# XPDSS Adaptive Stabilization Roadmap

## Visione
XPDSS deve diventare un **content-adaptive renderer affidabile**, non solo interessante sulla carta: l’analisi deve leggere il contenuto in tempo reale, il control plane deve produrre target sensati e la pipeline DSP deve applicarli **senza crash, senza telemetria ingannevole e senza comportamenti incoerenti tra stereo, surround, speaker e headphone**.

## Vincoli dichiarati
- **Niente misurazioni/calibrazioni binaurali** nel percorso operativo principale.
- Prima si stabilizza ciò che è già stato introdotto; nuove feature solo dopo la messa in sicurezza.
- Ogni target adaptive mostrato in UI deve avere un effetto reale e verificabile.
- L’adaptive non è considerato “fatto” finché non regge una prova runtime credibile, non solo test unitari.

---

## Stato attuale

### Completato
- Modulo analitico `adaptive_renderer.py` presente.
- Modulo di controllo `adaptive_control.py` presente.
- Stato profili `xpdss_state.py` presente.
- UI adaptive presente in `gui_components.py`.
- Test unitari dedicati presenti e verdi su WSL.
- Compile/import OK su WSL e Windows.

### Problemi confermati
- `Adaptive FULL` può andare in errore runtime con:
  - `ValueError sos must be shape (n_sections, 6)`
- L’analisi adaptive può non aggiornarsi su input stereo puro.
- Alcuni target adaptive sono mostrati in telemetria ma non hanno effetto DSP reale in tutte le condizioni.
- `center_extraction` adaptive non è applicato in modo uniforme tra i vari rami della pipeline.
- Stato/config non completamente allineati (`dialogue_focus`, `adaptive_mode`, integrazione `xpdss_state.py`).

---

## Fase A — Runtime Safety First
**Obiettivo:** impedire crash e fallback silenziosi nella pipeline adaptive.

### Deliverable
- Inizializzazione affidabile dei filtri SOS usati da clarity/room/air.
- Guard espliciti se un modulo adaptive non è pronto.
- `Adaptive FULL` sicuro anche con input stereo e configurazione reale.
- Nessun path adaptive che possa rompere `_apply_upmix()` per filtri non inizializzati.

### Success criteria
- Nessun crash nel probe runtime adaptive/full.
- Nessun `ValueError sos must be shape (n_sections, 6)`.
- I moduli non pronti fanno fallback sicuro invece di esplodere.

---

## Fase B — Stereo Analysis & Snapshot Integrity
**Obiettivo:** fare in modo che l’adaptive osservi davvero anche i contenuti stereo, non solo discrete surround.

### Deliverable
- Revisione di `_analysis_worker()` per evitare che stereo puro blocchi l’aggiornamento di `adaptive_snapshot`.
- Separazione chiara tra:
  - rilevamento multicanale discreto
  - disponibilità dell’analisi adaptive
- Snapshot aggiornati sia su stereo sia su multicanale.

### Success criteria
- `adaptive_snapshot` si aggiorna con sorgenti stereo normali.
- `is_multichannel_detected` non disabilita per errore l’intero cervello adaptive.

---

## Fase C — Real Target Application
**Obiettivo:** allineare telemetria, control plane e DSP reale.

### Deliverable
- Chiarezza su quali moduli l’adaptive può:
  - solo modulare se già accesi manualmente
  - oppure attivare direttamente in `assist/full`
- Applicazione coerente di target come:
  - `clarity_amount`
  - `virtual_room_mix`
  - `air_amount`
  - `micro_dyn_amount`
  - `transient_amount`
  - `dialogue_focus`
  - `vocal_lock`
  - `center_extraction`
  - `hp_externalization`
- Rimozione dei casi in cui la UI mostra target attivi ma il DSP fa `return` immediato.

### Success criteria
- Ogni target esposto in telemetria ha effetto reale o è esplicitamente dichiarato come observe-only.
- `assist` e `full` hanno differenze chiare e verificabili.

---

## Fase D — State / Config / GUI Coherence
**Obiettivo:** fare in modo che salvataggio, caricamento, UI e backend parlino la stessa lingua.

### Deliverable
- Allineamento di `curr_config`, `config.json` e `xpdss_state.py`.
- Presenza coerente di campi come `dialogue_focus`, `adaptive_mode`, `adaptive_profile`, eventuale compare bypass.
- GUI sincronizzata con il backend senza stati fantasma.

### Success criteria
- Save/load non perde campi adaptive.
- Le modalità adaptive sopravvivono a restart/import/export configurazione.

---

## Fase E — Integration Tests That Matter
**Obiettivo:** coprire i path che oggi i test unitari non stanno difendendo.

### Deliverable
- Test su `_apply_upmix()` con `observe`, `assist`, `full`.
- Test di non-regressione per filtri non inizializzati.
- Test che verifichino aggiornamento snapshot su stereo.
- Test che verifichino coerenza minima tra target adaptive e DSP applicato.

### Success criteria
- I test coprono i bug reali già osservati, non solo moduli isolati.
- Un refactor futuro non può reintrodurre silenziosamente i problemi attuali.

---

## Ordine operativo obbligatorio
1. **Mettere in sicurezza il runtime**.
2. **Ripristinare l’analisi adaptive su stereo**.
3. **Rendere i target realmente efficaci**.
4. **Pulire stato/config/UI**.
5. **Blindare con test di integrazione**.

---

## Definizione di completamento
Lavoro completato solo se:
- `Adaptive FULL` non crasha.
- L’analisi adaptive funziona su stereo e multicanale.
- La telemetria corrisponde al comportamento DSP reale.
- Stato/config/UI sono coerenti.
- Esistono test che difendono i bug corretti.
