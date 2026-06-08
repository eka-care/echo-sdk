# Health agent examples

Terminal-only examples for `health_agents`. Each script builds an agent,
runs it on a sample consultation transcript, and prints to stdout — no UI.

## Setup

Configure an LLM provider via environment (a `.env` at the repo root is loaded
automatically). Defaults to Bedrock; override with the standard echo vars:

```bash
ECHO_DEFAULT_LLM_PROVIDER=anthropic
ECHO_DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
# plus the provider's credentials (ANTHROPIC_API_KEY / AWS creds / OPENAI_API_KEY)
```

## Examples

| Script | What it shows |
|--------|---------------|
| `transcript_to_clinical_notes_example.py` | Non-streaming note: prints the full Markdown clinical note. |
| `transcript_to_clinical_notes_streaming_example.py` | Streaming note: prints the note as it is generated. |
| `transcript_to_clinical_notes_ag_ui_example.py` | AG-UI: prints each AG-UI event and the final structured `DocumentState` sections, built via the generic section tools. Requires the optional `ag_ui` dependency. |
| `ag_ui_generic_document_example.py` | The same generic AG-UI document tools used on **non-clinical** notes (a plain `AgUiAgent` structuring arbitrary text). Shows the tools are domain-agnostic. Requires `ag_ui`. |
| `docassist_mcp_example.py` | Fetches Eka Care MCP tools with a bearer token and lists them. **Gated** on `EKA_BEARER_TOKEN`; skips cleanly if unset. |

Run any of them from the repo root:

```bash
python examples/health_agents/transcript_to_clinical_notes_example.py
```

For the MCP example, supply the token whoever uses the SDK would pass:

```bash
EKA_BEARER_TOKEN=... python examples/health_agents/docassist_mcp_example.py
```

## Fixtures

- `fixtures/sample_transcript.txt` — a synthetic doctor–patient consultation.

The doctor template (the "user prompt") used by the examples is
`default_clinical_template`, bundled under
`src/echo/health_agents/prompts/user_prompts/`.
