#!/usr/bin/env bash
# Run from echo-sdk repo root:  bash "Claude outputs/echo_smoke.sh"
# Needs SARVAM_API_KEY and ELEVENLABS_API_KEY in echo-sdk/.env (or exported).
set -u
cd "$(dirname "$0")/.."
LOG="Claude outputs/echo_smoke.log"
: > "$LOG"
log() { echo "$@" | tee -a "$LOG"; }
run() { log ""; log "### $*"; "$@" 2>&1 | tee -a "$LOG"; log "### exit=${PIPESTATUS[0]}"; }

log "== 1. sync deps (adds elevenlabs + sarvamai extras, updates uv.lock) =="
run uv sync --extra sarvam --extra elevenlabs --extra dev

log ""
log "== 2. unit tests =="
run uv run pytest tests/test_streaming_transcribers.py tests/test_onprem_providers.py -q

log ""
log "== 3. ElevenLabs SDK realtime surface dump =="
run uv run python - <<'PY'
import inspect, pkgutil, elevenlabs, importlib
print("elevenlabs version:", getattr(elevenlabs, "__version__", "?"))
from elevenlabs import AsyncElevenLabs
c = AsyncElevenLabs(api_key="x")
stt = c.speech_to_text
print("speech_to_text attrs:", [a for a in dir(stt) if not a.startswith("_")])
rt = getattr(stt, "realtime", None)
print("realtime:", type(rt))
if rt is not None:
    for name in ("connect", "stream_url", "stream_file"):
        fn = getattr(rt, name, None)
        if fn:
            print(f"--- {name} {inspect.signature(fn)}")
    print("realtime module file:", inspect.getsourcefile(type(rt)))
# find the connection class
for modname in ("elevenlabs.realtime.connection", "elevenlabs.speech_to_text.realtime.connection", "elevenlabs.realtime.scribe"):
    try:
        m = importlib.import_module(modname)
        print("=== module", modname, m.__file__)
        for n, obj in inspect.getmembers(m, inspect.isclass):
            if obj.__module__ == modname:
                print("class", n, [a for a in dir(obj) if not a.startswith("_")])
                for meth in ("send", "send_audio", "commit", "close", "on", "__aiter__", "__anext__"):
                    f = getattr(obj, meth, None)
                    if f:
                        try: print("   ", meth, inspect.signature(f))
                        except Exception as e: print("   ", meth, "(sig n/a)")
    except Exception as e:
        print("no module", modname, e)
import os
base = os.path.dirname(elevenlabs.__file__)
for root, dirs, files in os.walk(base):
    for f in files:
        if "realtime" in root or "realtime" in f or "scribe" in f:
            p = os.path.join(root, f)
            if p.endswith(".py"):
                print("FILE", os.path.relpath(p, base))
PY

log ""
log "== 4. build a 16 kHz test WAV with macOS say =="
say -v Samantha -o "Claude outputs/smoke.aiff" "Patient complains of chest pain since two days. Blood pressure one forty over ninety. Advised Tab Metformin five hundred milligram twice daily." 2>&1 | tee -a "$LOG"
run afconvert -f WAVE -d LEI16@16000 -c 1 "Claude outputs/smoke.aiff" "Claude outputs/smoke16k.wav"

log ""
set -a; [ -f .env ] && source .env; set +a
# fall back to the keys in voice2rx-be/.env so they only need to be set once
if [ -z "${SARVAM_API_KEY:-}" ] || [ -z "${ELEVENLABS_API_KEY:-}" ]; then set -a; [ -f ../voice2rx-be/.env ] && source ../voice2rx-be/.env; set +a; fi
PROVIDERS="${PROVIDERS:-sarvam}"   # e.g. PROVIDERS="sarvam elevenlabs" to smoke both
log "providers under test: $PROVIDERS"
for prov in $PROVIDERS; do
  case $prov in sarvam) k=SARVAM_API_KEY;; elevenlabs) k=ELEVENLABS_API_KEY;; esac
  [ -n "${!k:-}" ] || log "!! $k is not set - add it to echo-sdk/.env or voice2rx-be/.env before the live smokes"
done

for prov in $PROVIDERS; do
  log ""
  log "== 5. live smoke: $prov =="
  run uv run python examples/streaming_transcription_usage.py "Claude outputs/smoke16k.wav" $prov en
done

log ""
log "== done: $(date) =="
