# Real-time Dictation — Implementation Plan (echo-sdk + voice2rx-be)

**Goal.** Let a doctor press the mic button (or the global shortcut) in the EkaScribe desktop app and dictate anywhere; the backend streams audio to a speech-to-text provider (Sarvam or ElevenLabs, via their official SDKs) and streams text back in real time.

**Split of responsibility.**

| Layer | Repo | Owns |
|---|---|---|
| Provider abstraction + SDK calls | `echo-sdk` (`src/echo/audio/transcription/`) | `BaseStreamingTranscriber`, `SarvamStreamingTranscriber`, `ElevenLabsStreamingTranscriber`, streaming event schema, factory + extras |
| Product API | `voice2rx-be` (`voice2rx/dictation/`) | `POST /voice/v1/dictation/sessions`, `WS /voice/v1/dictation/sessions/{id}/audio`, provider/language selection, session store, transcript persistence, usage metering |
| Client | DeskDocEka / vaarta-desktop (out of scope here) | mic capture → 16 kHz PCM → WS; render partial/final text; insert at cursor |

Both repos already have the right scaffolding: echo-sdk has a batch `SarvamTranscriber` on `sarvamai`, and voice2rx-be has a Redis-backed WebSocket streaming module (`voice2rx/streaming/`) whose shape we copy. Nothing in the existing batch transcription or VAD-chunking pipeline is touched.

---

## 0. Findings from the code that shape the design

**echo-sdk (`main`, v0.4.3)**

- `audio/transcription/` already has `BaseTranscriber.transcribe()` (batch only), `TranscriberConfig(provider: Literal["gemini","ekacare","sarvam"])`, `get_transcriber()` factory, and `sarvam.py` using `AsyncSarvamAI` (`sarvamai` 0.1.28 is in the venv). Repo rules (CLAUDE.md + `.claude/skills`): factory-only, optional deps guarded inside methods, `orjson` only, Pydantic v2, async-first, don't change `BaseX` to fit a provider.
- `sarvamai` 0.1.28 already ships a streaming client: `client.speech_to_text_streaming.connect(language_code=..., model="saaras:v3"|"saarika:v2.5", sample_rate="16000", input_audio_codec="pcm_s16le", vad_signals="true", flush_signal="true", high_vad_sensitivity=...)` → async context manager yielding `AsyncSpeechToTextStreamingSocketClient` with `await socket.transcribe(audio=<base64>, encoding="audio/wav", sample_rate=16000)`, `await socket.flush()`, and `async for msg in socket` where `msg.type ∈ {"data","error","events"}` and `msg.data.transcript` / `.language_code` / `.request_id`. Sarvam emits **per-utterance finals only** (VAD-segmented), no interim partials.
- ElevenLabs realtime STT (Scribe v2 Realtime): Python SDK `elevenlabs` — `async with client.speech_to_text.realtime.connect(model_id="scribe_v2_realtime", audio_format="pcm_16000", commit_strategy="vad"|"manual", language_code=..., include_timestamps=..., keyterms=[...]) as connection:` then `await connection.send({"audio_base_64": b64, "sample_rate": 16000})`, `await connection.commit()`, and events via `async for event in connection` / `connection.on(...)` with `message_type ∈ {session_started, partial_transcript, committed_transcript, committed_transcript_with_timestamps, error, rate_limited, ...}`. Emits **partials + committed finals**. ⚠️ The SDK surface was verified from ElevenLabs docs, not from an installed wheel (PyPI is blocked from this sandbox) — step 1.7 below pins this down before writing the provider.

**voice2rx-be (`master`)**

- FastAPI; routers in `main.py`; `voice2rx/streaming/` already implements the pattern we need: `stream_session_router` (HTTP creates session, returns `wss_url`), `stream_ws_router` (binary PCM or JSON-envelope frames), `stream_session_store` (Redis, `stream:session:{id}`), `_finalize()` on disconnect.
- Auth: gateway injects a `jwt-payload` header (`b-id`, `uuid`, `c-id`); `RequestHandler.extract_token_data_from_request()` and `voice2rx/api/dependencies.py:get_validated_jwt_payload`. WebSockets can't carry that header from a browser/Electron renderer, so the existing design uses the server-minted `stream_id` as the WS capability — we keep that.
- `echo` arrives transitively through `echo-agents` (private `eka-care/echo`, pinned `v0.3.12`). Extras don't propagate, so voice2rx-be must declare `sarvamai` and `elevenlabs` directly (it already does this for `openai`, `anthropic`, `google-genai`).
- Config resolution precedence used elsewhere: business config (`ConfigService.get_merged_config`) → env → defaults.
- nginx (`dockerconfig/vhost.conf`) only upgrades WebSockets for `location /voice/v1/stream/` (3600 s read timeout); the default location has a 300 s timeout and no `Upgrade` headers. A new `/voice/v1/dictation/` location is required.
- Tests: `tests/conftest.py` injects `sys.modules` mocks for native deps; `main` must stay import-safe. `choices.py` already has `TransactionMode.DICTATION` and `ASRService.ELEVENLABS` (unused).
- `logs.custom_logger` structured logging with `severity="critical|medium|low"` (see `voice2rx/LOG_CRITICALITY.md`); usage metering via `voice2rx/utils/eka_usage_client.record_safe`.

---

## 1. echo-sdk changes

Target version: **0.5.0** (new public API surface).

### 1.1 `src/echo/audio/transcription/schemas.py` — streaming event contract

Add (keep existing `AudioInput`, `TokenUsage`, `TranscriptionResponse` untouched):

```python
class StreamingEventType(str, Enum):
    SESSION_STARTED = "session_started"
    PARTIAL = "partial"          # interim, may be revised (ElevenLabs only)
    FINAL = "final"              # committed segment (both providers)
    SPEECH_STARTED = "speech_started"  # provider VAD signal, optional
    SPEECH_ENDED = "speech_ended"
    ERROR = "error"
    DONE = "done"                # provider closed the stream cleanly

class StreamingTranscriptEvent(BaseModel):
    type: StreamingEventType
    text: Optional[str] = None
    segment_id: Optional[str] = None      # provider request_id / segment counter
    language_detected: Optional[str] = None
    timestamps: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None   # raw provider fields (metrics, probability)
    error: Optional[str] = None
    recoverable: bool = True     # False → caller should end the session
```

This mirrors `llm/schemas.py:StreamEvent` (typed enum + optional payload fields) so consumers get one familiar shape.

### 1.2 `src/echo/audio/transcription/config.py` — streaming options

- Extend the `provider` Literal to `["gemini", "ekacare", "sarvam", "elevenlabs"]`.
- Add streaming fields to `TranscriberConfig` (defaults chosen so batch callers are unaffected):

```python
sample_rate: int = 16000                 # streaming input rate; both SDKs prefer 16 kHz
encoding: Literal["pcm_s16le"] = "pcm_s16le"  # raw 16-bit mono PCM; only format both accept
commit_strategy: Literal["vad", "manual"] = "vad"
vad_silence_threshold_s: Optional[float] = None
high_vad_sensitivity: bool = False
interim_results: bool = True             # ElevenLabs partials on/off
keyterms: Optional[List[str]] = None     # ElevenLabs keyterm biasing (≤50; 20 % cost premium)
mode: Optional[Literal["transcribe","verbatim","translate","translit","codemix"]] = None  # Sarvam saaras:v3
```

- Model defaulting in `_validate_model`: when `provider == "elevenlabs"` and the model is gemini-shaped, set `os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2_realtime")`. For Sarvam keep `saarika:v2.5` as the batch default but let the streaming transcriber read `SARVAM_STT_STREAM_MODEL` (default `saaras:v3`, the recommended streaming model) when the configured model is a batch-only name.
- Add `STREAMING_PROVIDERS = ("sarvam", "elevenlabs")` constant used by the factory error message.

### 1.3 `src/echo/audio/transcription/base.py` — streaming abstraction

Do **not** modify `BaseTranscriber` (repo rule). Add two new abstracts alongside it:

```python
class StreamingTranscriptionSession(ABC):
    """One open provider websocket. Obtained via BaseStreamingTranscriber.stream()."""
    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...
    @abstractmethod
    async def flush(self) -> None: ...          # force-commit buffered audio
    @abstractmethod
    async def close(self) -> None: ...          # idempotent
    @abstractmethod
    def events(self) -> AsyncIterator[StreamingTranscriptEvent]: ...
    def __aiter__(self): return self.events()

class BaseStreamingTranscriber(ABC):
    def __init__(self, config: TranscriberConfig): ...
    @abstractmethod
    def stream(self) -> AsyncContextManager[StreamingTranscriptionSession]: ...
```

Contract notes to put in the docstrings: `send_audio` takes raw `pcm_s16le` bytes at `config.sample_rate` (the provider class does the base64 wrapping); `events()` terminates after `DONE` or a non-recoverable `ERROR`; `close()` must be safe to call from `finally`; callers own chunk sizing (recommend 100–250 ms frames, i.e. 3 200–8 000 bytes at 16 kHz).

### 1.4 `src/echo/audio/transcription/sarvam.py` — `SarvamStreamingTranscriber`

Same file as the batch class (shares `_LANGUAGE_MAP`, client construction, `base_url` override).

- `stream()` → `@asynccontextmanager` that opens `self.client.speech_to_text_streaming.connect(language_code=self._language_code(), model=<stream model>, sample_rate=str(config.sample_rate), input_audio_codec="pcm_s16le", vad_signals="true", flush_signal="true", high_vad_sensitivity="true" if config.high_vad_sensitivity else None, mode=config.mode)` and yields `_SarvamSession(socket)`.
- `_SarvamSession.send_audio(pcm)` → `await socket.transcribe(audio=base64.b64encode(pcm).decode(), encoding="audio/wav", sample_rate=config.sample_rate)` (the SDK's `AudioData.encoding` literal is `"audio/wav"` even for raw PCM — the codec is declared at connect time via `input_audio_codec`).
- `flush()` → `await socket.flush()`.
- `events()` → `async for msg in socket:` map `type=="data"` → `FINAL` (`text=msg.data.transcript`, `segment_id=msg.data.request_id`, `language_detected=msg.data.language_code`, `details={"language_probability", "metrics"}`), `type=="events"` → `SPEECH_STARTED`/`SPEECH_ENDED` from the VAD signal payload, `type=="error"` → `ERROR` (`recoverable=False` on auth/quota codes), and emit `DONE` when the iterator ends. Wrap `websockets.ConnectionClosed` → `DONE`, other exceptions → `ERROR(recoverable=False)`.
- `close()` → close the underlying websocket (`socket._websocket.close()` — note the SDK does not expose a public close; the context-manager exit handles it, so `close()` just cancels the reader and exits).
- Keep Sarvam's constraints in the docstring: connection-level `sample_rate` must be 16000 or 8000; no interim partials.

### 1.5 `src/echo/audio/transcription/elevenlabs.py` — new file

- Batch `ElevenLabsTranscriber(BaseTranscriber)` using `AsyncElevenLabs().speech_to_text.convert(model_id=os.getenv("ELEVENLABS_STT_BATCH_MODEL","scribe_v1"), file=..., language_code=...)` — small, keeps the factory symmetric (`get_transcriber(provider="elevenlabs")`). Optional; can be a follow-up PR if scope is tight.
- `ElevenLabsStreamingTranscriber(BaseStreamingTranscriber)`:
  - API key from `config.api_key or ELEVENLABS_API_KEY`; lazy `AsyncElevenLabs` inside the property (guarded import → `ImportError("elevenlabs is required … pip install 'echo-sdk[elevenlabs]'")`).
  - `stream()` → `async with self.client.speech_to_text.realtime.connect(model_id=self.model, audio_format=f"pcm_{config.sample_rate}", commit_strategy=config.commit_strategy, language_code=<ISO-639-1 from config.language, None → auto-detect>, vad_silence_threshold_secs=config.vad_silence_threshold_s, include_timestamps=False, keyterms=config.keyterms) as conn: yield _ElevenLabsSession(conn)`.
  - `send_audio(pcm)` → `await conn.send({"audio_base_64": base64…, "sample_rate": config.sample_rate})`.
  - `flush()` → `await conn.commit()`.
  - `events()` → iterate `conn`; map `partial_transcript` → `PARTIAL` (only if `config.interim_results`), `committed_transcript` / `committed_transcript_with_timestamps` → `FINAL`, `session_started` → `SESSION_STARTED` (`details=config echo`), `rate_limited` → `ERROR(recoverable=True)`, `scribeAuthError`/`scribeQuotaExceededError`/`scribeInvalidRequestError`/`scribeResourceExhaustedError` → `ERROR(recoverable=False)`.
  - Language map: ElevenLabs takes ISO-639-1/-3 (`hi`, `en`, `ta`…); echo callers pass `"hi"`/`"en"` already, and Sarvam-style `hi-IN` should be trimmed to `hi`.

### 1.6 `src/echo/audio/transcription/factory.py` and `__init__.py`

- New `get_streaming_transcriber(config: TranscriberConfig) -> BaseStreamingTranscriber` with `sarvam` and `elevenlabs` branches, same lazy-import + install-hint pattern as `get_transcriber`; raise `ValueError("Streaming not supported for provider 'gemini'/'ekacare'. Supported: sarvam, elevenlabs")` otherwise.
- Add the `elevenlabs` branch to `get_transcriber` (if 1.5 batch is included).
- Export `BaseStreamingTranscriber`, `StreamingTranscriptionSession`, `StreamingTranscriptEvent`, `StreamingEventType`, `get_streaming_transcriber` from `transcription/__init__.py` and `audio/__init__.py`.

### 1.7 `pyproject.toml`

```toml
sarvam     = ["sarvamai>=0.1.28"]          # bump: streaming client + input_audio_codec arrived after 0.1.20
elevenlabs = ["elevenlabs>=2.19.0"]        # ⚠ verify: first version exposing speech_to_text.realtime; pin after `uv add elevenlabs` and inspecting src/elevenlabs/realtime/
all        = [..., "sarvamai>=0.1.28", "elevenlabs>=2.19.0"]
```

`websockets` comes in transitively from both SDKs; don't declare it. Bump `version = "0.5.0"`. Run `uv lock`.

First implementation step for 1.5: `uv add --optional elevenlabs elevenlabs`, then read `.venv/lib/python*/site-packages/elevenlabs/realtime/{scribe,connection}.py` and `speech_to_text_custom.py` and adjust the method names in 1.5 to match the installed SDK (two documented shapes exist: `connection.send(dict)` + `async for event in connection`, and `connection.on("PARTIAL_TRANSCRIPT", handler)`; the iterator form is preferred because it fits `events()` directly).

### 1.8 Tests — `tests/test_streaming_transcribers.py`

No network. Fake socket objects with an `async for` of canned messages:

- Factory: `get_streaming_transcriber(TranscriberConfig(provider="sarvam"))` returns `SarvamStreamingTranscriber`; `provider="gemini"` raises `ValueError`; missing SDK raises `ImportError` with the install hint (monkeypatch `sys.modules`).
- Sarvam mapping: `{"type":"data","data":{"transcript":"नमस्ते","request_id":"r1","language_code":"hi-IN",…}}` → one `FINAL` event; `{"type":"error",…}` → `ERROR`; iterator exhaustion → `DONE`; `send_audio(b"\x00\x01")` calls `socket.transcribe(audio="AAE=", encoding="audio/wav", sample_rate=16000)`; `flush()` calls `socket.flush()`.
- ElevenLabs mapping: partial → `PARTIAL`, committed → `FINAL`, `interim_results=False` suppresses partials, `rate_limited` → recoverable, auth error → non-recoverable; `flush()` → `conn.commit()`.
- Config: `provider="elevenlabs"` defaults model to `scribe_v2_realtime`; language normalisation (`hi-IN`→`hi` for ElevenLabs, `hi`→`hi-IN` for Sarvam).
- Extend `tests/test_onprem_providers.py::test_sarvam_provider_registered` unchanged; add one asserting the streaming model default.

### 1.9 Docs / examples

- `examples/streaming_transcription_usage.py`: read a 16 kHz WAV, send 100 ms frames, print partial/final events, switch provider via `ECHO_DEFAULT_TRANSCRIBER_PROVIDER`.
- `.claude/skills/echo-sdk/audio/SKILL.md`: document the streaming abstraction, the "bytes in, events out" rule and that only sarvam/elevenlabs stream.
- `README.md` install matrix: add `echo-sdk[elevenlabs]`, note `echo-sdk[sarvam]` now also covers streaming.
- `.env.sample`: `SARVAM_API_KEY`, `ELEVENLABS_API_KEY`, `SARVAM_STT_STREAM_MODEL`, `ELEVENLABS_STT_MODEL`.

### 1.10 Release chain (easy to forget)

echo-sdk `v0.5.0` tag → re-pin echo-sdk inside the private `eka-care/echo` (`echo-agents`) `[tool.uv.sources]` and tag it (e.g. `v0.3.13`) → bump the `echo-agents` tag in voice2rx-be `pyproject.toml` → `uv lock`. While developing, use the commented `path = "../../echo", editable = true` source in voice2rx-be (don't commit).

---

## 2. voice2rx-be changes

New package `voice2rx/dictation/` mirroring `voice2rx/streaming/` (api / session / services). Nothing under `streaming/`, `protocol/` or `services/transactions/` changes.

### 2.1 API surface

**`POST /voice/v1/dictation/sessions`** (`voice2rx/dictation/api/dictation_session_router.py`)

Request (all optional; identity comes from the `jwt-payload` header via `RequestHandler.extract_token_data_from_request`, body fallback for M2M like the stream router):

```json
{ "provider": "sarvam" | "elevenlabs" | null,
  "language": "hi" | "en" | null,
  "keyterms": ["Metformin", "Amlodipine"],
  "session_id": "<optional existing scribe session to attach the transcript to>",
  "client": { "app": "deskdoc", "version": "1.4.0", "trigger": "shortcut" | "mic_button" } }
```

Response: `{ "dictation_id": "dct_…", "wss_url": "wss://<HOST>/voice/v1/dictation/sessions/dct_…/audio", "provider": "sarvam", "model": "saaras:v3", "language": "hi", "audio": { "encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1, "frame_ms": 100 }, "expires_in_s": 600 }`.

The `audio` block tells the client exactly what to send so the desktop app never guesses.

**`WS /voice/v1/dictation/sessions/{dictation_id}/audio`** (`voice2rx/dictation/api/dictation_ws_router.py`)

Client → server frames:
- binary: raw `pcm_s16le` 16 kHz mono, 100–250 ms per frame;
- text JSON: `{"event":"start","mediaFormat":{"sampleRate":16000}}` (optional), `{"event":"flush"}` (force-commit, e.g. user paused), `{"event":"stop"}` (graceful end; server flushes, drains finals, sends `done`, closes).

Server → client frames (text JSON, one per event):

```json
{"type":"session_started","dictation_id":"dct_…","provider":"sarvam"}
{"type":"partial","text":"patient complains of","seq":12}
{"type":"final","text":"Patient complains of chest pain since two days.","segment_id":"r7","seq":13,"language":"en-IN"}
{"type":"error","code":"provider_rate_limited","message":"…","recoverable":true}
{"type":"done","transcript":"<full text of all finals joined>","segments":14,"duration_s":42.3}
```

Close codes: `4000` unknown/expired dictation_id, `4001` session already in use, `4008` idle timeout, `4011` provider failure after retry, `1000` normal.

**`GET /voice/v1/dictation/sessions/{dictation_id}`** — status + transcript so far (lets the client recover text after a dropped connection; also useful for QA).

**`DELETE /voice/v1/dictation/sessions/{dictation_id}`** — abort (client cancelled before connecting).

### 2.2 Files

| Path | Purpose |
|---|---|
| `voice2rx/dictation/__init__.py` | package marker |
| `voice2rx/dictation/api/__init__.py` | exports `dictation_session_router`, `dictation_ws_router` |
| `voice2rx/dictation/api/dictation_session_router.py` | HTTP: create / get / delete session. Resolves `b_id`/`uuid`, calls `DictationConfigResolver`, mints `dictation_id = "dct_" + secrets.token_hex(10)`, saves to Redis with `status="created"`, builds `wss_url` from `HOST` (same as `stream_session_router`). |
| `voice2rx/dictation/api/dictation_ws_router.py` | WS handler: accept → load session (reject 4000 if missing; 4001 if `status=="streaming"`) → `DictationBridge.run(ws)` → `finally: finalize`. Thin — all logic in the bridge. |
| `voice2rx/dictation/session/dictation_session_store.py` | Redis: `dictation:session:{id}` (metadata JSON, TTL 600 s until WS connects, then 2 h), `dictation:segments:{id}` (RPUSH of final segments as `{"seq","text","segment_id","ts"}`). Same API shape as `StreamSessionStore` (`save/get/update/delete`, `append_segment`, `get_segments`). Separate class rather than generalising `StreamSessionStore` — keeps the streaming module untouched (small-diffs rule); revisit if a third stream type appears. |
| `voice2rx/dictation/services/dictation_config.py` | `DictationConfigResolver.resolve(b_id, uuid, request_overrides) -> DictationConfig` with precedence request body → `ConfigService.get_merged_config(b_id, uuid).get("dictation", {})` (`{"stt_provider","language","model","keyterms"}`) → env (`DICTATION_STT_PROVIDER` default `sarvam`, `DICTATION_STT_MODEL`, `DICTATION_LANGUAGE` default auto) → defaults. Produces the echo `TranscriberConfig` (`provider`, `model`, `language`, `sample_rate=16000`, `commit_strategy="vad"`, `keyterms`, `api_key` from `SARVAM_API_KEY`/`ELEVENLABS_API_KEY`). Also exposes the `audio` contract block returned to the client. |
| `voice2rx/dictation/services/dictation_bridge.py` | The core. `DictationBridge(ws, dictation_id, session_meta, transcriber)` runs two tasks under `asyncio.TaskGroup`: **uplink** (`ws.receive()` → binary → `session.send_audio`; `flush`/`stop` events; JSON `start` sets sample rate if ≠16000 → reject with 4002 since we don't resample) and **downlink** (`async for ev in session` → map to wire JSON → `ws.send_text`; on `FINAL` also `store.append_segment` and accumulate in memory). Handles: idle timeout (`DICTATION_IDLE_TIMEOUT_S`, default 60 s of no audio → flush + close 4008), max duration (`DICTATION_MAX_DURATION_S`, default 1800), provider `ERROR(recoverable=True)` → surface to client and keep going, non-recoverable → one reconnect attempt with a fresh provider session (audio sent during the gap is dropped, client gets `{"type":"error","code":"provider_reconnected"}`), second failure → close 4011. Backpressure: bounded `asyncio.Queue(maxsize=50)` between uplink and provider; if full, drop oldest and log `severity="medium"`. Stop sequence: `flush()` → wait ≤ `DICTATION_DRAIN_TIMEOUT_S` (3 s) for trailing finals → `done` frame → `close(1000)`. |
| `voice2rx/dictation/services/dictation_service.py` | Orchestration that the routers call: `create_session`, `get_session`, `finalize(dictation_id, transcript, duration_s, reason)`. Finalize writes the transcript + stats back to Redis (status `completed`/`failed`), optionally patches the linked scribe transaction (`session_id` present → `TransactionService.update_transaction(session_id, b_id, {"dictation_transcript": …})`; keep behind `DICTATION_ATTACH_TO_SESSION=true` until product confirms the field), and meters usage via `record_safe(workspace_id=b_id, product="ekascribe", metric_type="dictation_seconds", quantity=duration_s, metadata={"provider","model","language"})`. |
| `voice2rx/dictation/schemas.py` | Pydantic request/response models for the HTTP routes and the wire-event models (`DictationWireEvent`) so the WS JSON is typed and testable. |
| `main.py` | `from voice2rx.dictation.api import dictation_session_router, dictation_ws_router` and `app.include_router(…, prefix="/voice/v1/dictation", tags=["dictation"])` ×2, next to the streaming routers. |
| `pyproject.toml` | add `"sarvamai>=0.1.28"`, `"elevenlabs>=2.19.0"` (extras don't propagate through `echo-agents`); bump `echo-agents` tag after 1.10; `uv lock`. |
| `dockerconfig/vhost.conf` | new `location /voice/v1/dictation/ { … proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; proxy_read_timeout 3600s; proxy_buffering off; }` cloned from the `/voice/v1/stream/` block. Without it the WS upgrade dies at nginx. |
| `.env` / secrets (`voice2rx` JSON blob in Secrets Manager) | `SARVAM_API_KEY`, `ELEVENLABS_API_KEY`, `DICTATION_STT_PROVIDER`, `DICTATION_STT_MODEL`, `DICTATION_IDLE_TIMEOUT_S`, `DICTATION_MAX_DURATION_S`, `DICTATION_ATTACH_TO_SESSION`. Also the outbound egress must allow `api.sarvam.ai` / `wss://api.sarvam.ai` and `wss://api.elevenlabs.io`. |
| `voice2rx/openapi/v1_openapi.json` | add the three HTTP routes (WS documented in the subsystem doc). |
| `.claude/docs/subsystems/dictation.md` + `INDEX.md` row + `tech-debt.md` section | required by the repo's sync rule. |
| `voice2rx/choices.py` | add `ASRService.SARVAM = "SARVAM"`; reuse `ELEVENLABS`. |

### 2.3 Session lifecycle

```
Client (mic/shortcut) ──POST /dictation/sessions──▶ router ─▶ resolver ─▶ Redis(status=created, ttl 600s)
        ◀── {dictation_id, wss_url, audio contract} ──
Client ──WS connect──▶ ws_router ─▶ Redis(status=streaming, ttl 2h)
        ─▶ get_streaming_transcriber(cfg).stream()  (echo-sdk → Sarvam/ElevenLabs websocket)
   PCM frames ──▶ bridge.uplink ──▶ session.send_audio
   {partial|final} ◀── bridge.downlink ◀── session.events()   (finals also RPUSH'd to Redis)
Client ──{"event":"stop"}──▶ flush → drain → {"type":"done", transcript} → close(1000)
                              └─▶ dictation_service.finalize → Redis(status=completed), usage metered,
                                  optional attach to scribe transaction
```

Disconnect without `stop` follows the same finalize path with `reason="disconnect"`; the client can then `GET` the transcript.

### 2.4 Tests (`tests/unit/`)

- `api/test_dictation_session_api.py` — create with jwt-payload header, create with body fallback, missing `b_id` → 400, provider override validation (`"whisper"` → 422), business-config precedence (mock `ConfigService`), response `audio` block, `GET` after finals, `DELETE`.
- `api/test_dictation_ws.py` — Starlette `TestClient.websocket_connect`; inject a `FakeStreamingTranscriber` (yields scripted `StreamingTranscriptEvent`s, records `send_audio` bytes) via a module-level factory hook (`dictation_bridge.get_streaming_transcriber` monkeypatched). Cases: unknown id → 4000; happy path binary frames → partial + final frames → `stop` → `done` with joined transcript; `flush` event calls `session.flush()`; provider non-recoverable error → reconnect once → second failure → 4011; idle timeout → 4008 (use a 0.1 s override); mismatched sample rate in `start` → 4002.
- `services/test_dictation_bridge.py` — queue backpressure drop, drain timeout, segment ordering (`seq` monotonic).
- `services/test_dictation_service.py` — finalize meters `dictation_seconds` and only patches the transaction when `DICTATION_ATTACH_TO_SESSION` is on.
- `conftest.py` — nothing native to mock (`sarvamai`/`elevenlabs` are pure Python), but add `sys.modules` stubs for both so the suite runs on machines without the extras, mirroring the `pipecat` pattern.
- `make run_test` must stay green; `main` import-safe.

### 2.5 Observability & ops

- Structured logs with `severity` per `LOG_CRITICALITY.md`: session created/completed (`medium` info), provider error/reconnect (`medium`), finalize failure or usage-metering failure (`critical`), dropped frames (`medium`).
- Metrics worth emitting in log fields: time-to-first-final, provider round-trip per segment (Sarvam `metrics` block), frames dropped, session duration, reconnects.
- Gunicorn: long WS connections hold a worker slot; with 4 uvicorn workers this is fine for async I/O but set `DICTATION_MAX_DURATION_S` and consider raising `--graceful-timeout` so deploys don't cut dictations mid-sentence (nginx already 3600 s).
- Gateway (api.eka.care): the `/voice/v1/dictation/*` prefix (including WS upgrade) must be routed the same way `/voice/v1/stream/*` is.

---

## 3. Client contract (for the desktop team — not implemented here)

Mic button / global shortcut → `POST /voice/v1/dictation/sessions` → open `wss_url` → capture mic at 16 kHz mono, convert to `pcm_s16le`, send 100 ms binary frames (3 200 bytes) → render `partial` as grey provisional text, replace with `final` (append + space) → on release/stop send `{"event":"stop"}`, wait for `done`, insert `done.transcript` (or the accumulated finals) at the cursor. On WS drop, `GET` the session to recover finals. Both providers produce far better results when the client sends exactly 16 kHz PCM rather than compressed audio, which is why resampling is a client responsibility.

---

## 4. Delivery order

1. **echo-sdk PR 1** — schemas, base, config, factory, `SarvamStreamingTranscriber`, tests, example. Verify live against Sarvam with a WAV. (Sarvam is the default provider; ships value alone.)
2. **echo-sdk PR 2** — `uv add elevenlabs`, confirm the realtime SDK surface, `ElevenLabsStreamingTranscriber` (+ optional batch), tests. Tag `v0.5.0`; re-pin in `eka-care/echo`.
3. **voice2rx-be PR 1** — `voice2rx/dictation/` module, `main.py`, nginx, env, tests, subsystem doc. Local dev with `echo-agents = { path = …, editable = true }`.
4. **voice2rx-be PR 2** — bump `echo-agents` tag, `uv lock`, secrets in dev/stage, egress allow-list, gateway route. Stage soak with the desktop app.
5. Follow-ups (not blocking): optional LLM "polish" pass over the final transcript (punctuation, drug-name normalisation) via the existing `GenericAgent` in `voice2rx/agents/` before `done`; per-workspace provider routing UI in business config; word timestamps for the editor.

---

## 5. Open decisions to confirm

1. **Default provider**: Sarvam (`saaras:v3`, best Indic coverage, no partials) vs ElevenLabs (`scribe_v2_realtime`, partials, ~30+ languages). Plan assumes Sarvam default, ElevenLabs opt-in per workspace/request.
2. **Attach dictation to a scribe session?** Plan supports an optional `session_id` link behind a flag; the field name on the transaction record needs a product decision.
3. **Usage metering unit**: `dictation_seconds` per session (plan) vs per-segment events.
4. **Interim partials on the wire** for Sarvam: none available; the client should show a "listening…" state instead. Confirm the UX is fine with segment-level updates (~1–3 s cadence).
5. **ElevenLabs SDK minimum version** — resolved during echo-sdk PR 2 (step 1.7).
