# Codex Auto Review structured-output handover

## User-visible failure

Codex users routed through `https://aaii.xclaw.info/v1/responses` reported:

```text
Automatic approval review failed:
guardian assessment was not valid JSON

stream disconnected before completion
```

## Root cause

Codex 0.144.6 uses the `codex-auto-review` model for automatic approval review.
The bundled Codex model catalog marks this model with `use_responses_lite: false`.
Guardian requests set `final_output_json_schema`, which Codex serializes into the
Responses API request as:

```json
{
  "text": {
    "format": {
      "type": "json_schema",
      "name": "codex_output_schema",
      "strict": false,
      "schema": {}
    }
  }
}
```

`ChatGPTResponsesAPIConfig.transform_responses_api_request()` builds the request
through the OpenAI Responses transformation and then filters it through a
ChatGPT-backend allowlist. That allowlist omitted `text`, so LiteLLM silently
removed the guardian JSON schema before forwarding the request. The model could
then return ordinary text, which Codex rejected as an invalid guardian
assessment.

The removal of top-level `instructions` for requests carrying
`x-openai-internal-codex-responses-lite` is intentional and is not the cause:
Responses Lite places developer instructions in `input` items, and
`codex-auto-review` is not a Responses Lite model.

## Changes in this branch

- Preserve the Responses API `text` parameter in the ChatGPT backend allowlist.
- Add a regression test that models a `codex-auto-review` guardian JSON-schema
  request and asserts that `text` reaches the provider request unchanged.

## Validation status

Tests were intentionally not run at the user's request. The workspace also has
no repository `.venv`, system `pytest`, or `python` executable; only `python3`
is present and it does not have `pytest` installed.

Recommended targeted validation for the next agent in a configured environment:

```bash
.venv/bin/pytest tests/test_litellm/llms/chatgpt/responses/test_chatgpt_responses_transformation.py -q
```

Recommended deployment validation:

1. Route `codex-auto-review` to the ChatGPT provider deployment.
2. Capture the outbound provider request and verify `text.format.type` is
   `json_schema`.
3. Trigger a Codex automatic approval review and verify the output contains at
   least `{"outcome":"allow"}` or a valid deny assessment.
4. Confirm the SSE stream contains a terminal `response.completed` event. If
   the separate disconnect persists after the JSON fix, inspect reverse-proxy
   buffering/timeouts and provider-stream logs; an EOF before
   `response.completed` is independently reported by Codex as
   `stream disconnected before completion`.

## Reference used during diagnosis

The behavior was checked against the official `openai/codex` source at the
current `main` revision on 2026-07-21, particularly:

- `codex-rs/model-provider/src/provider.rs`
- `codex-rs/models-manager/models.json`
- `codex-rs/core/src/client.rs`
- `codex-rs/core/src/guardian/prompt.rs`
- `codex-rs/core/src/guardian/review_session.rs`
