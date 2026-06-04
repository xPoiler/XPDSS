# XPDSS Adaptive Stabilization Plan

> Per eseguire usare workflow Planner / Coder / Reviewer in loop fino ad approvazione.

## Ordine esecutivo
1. Runtime safety
2. Analisi adaptive su stereo
3. Applicazione reale dei target adaptive
4. Uniformazione `center_extraction`
5. Coerenza stato/config/UI
6. Test e probe finale

## Task 1 — Runtime safety
- File: `audio_analyzer.py`
- Mettere in sicurezza init/validazione filtri SOS.
- Resettare gli state/ZI quando i filtri adaptive vengono reinizializzati.
- Fare fallback sicuro se un filtro non è pronto.
- Verifica:
  - probe Windows adaptive/full senza `ValueError sos must be shape (n_sections, 6)`
  - WSL `PYTHONPATH=. python -m pytest -q`

## Task 2 — Analisi stereo
- File: `audio_analyzer.py`
- Separare rilevamento multicanale da aggiornamento adaptive snapshot.
- Su `chs == 2` aggiornare comunque `adaptive_snapshot` / `adaptive_state`.
- Verifica:
  - probe Windows con contenuto stereo
  - snapshot adaptive non vuoto/stale

## Task 3 — Target adaptive reali
- File: `adaptive_control.py`, `audio_analyzer.py`, `gui_components.py`, `tests/test_adaptive_control.py`
- Allineare target controller e path DSP reali.
- Decidere semantica coerente per `assist` vs `full`.
- Sistemare mismatch tra target booleani adaptive e moduli realmente attivabili.
- Verifica:
  - test WSL aggiornati
  - probe Windows con `observe/assist/full`

## Task 4 — Uniformare center extraction
- File: `audio_analyzer.py`, opzionale `gui_components.py`
- Stessa semantica tra branch STFT e branch legacy/static.
- Rendere chiari i no-op inevitabili.
- Verifica:
  - probe Windows speaker mode

## Task 5 — Stato/config/UI coerenti
- File: `audio_analyzer.py`, `xpdss_state.py`, `gui_components.py`, `tests/test_xpdss_state.py`
- Ridurre duplicazione dello schema di stato.
- Allineare save/load e UI con i campi adaptive.
- Verifica:
  - test WSL round-trip stato
  - restart/save-load su Windows

## Task 6 — Chiusura con test e probe
- File: test esistenti + eventuali helper nuovi
- Difendere i bug reali corretti.
- Verifica:
  - WSL `PYTHONPATH=. python -m pytest -q`
  - probe Windows finale

## Reviewer checklist
- Nessun crash adaptive/full.
- Snapshot adaptive aggiornato su stereo.
- Telemetria coerente col DSP reale.
- `center_extraction` coerente nei branch.
- Stato/config/UI coerenti.
- Differenze chiare tra `observe`, `assist`, `full`.
