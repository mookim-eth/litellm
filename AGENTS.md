# INSTRUCTIONS FOR LITELLM

This document provides comprehensive instructions for AI agents working in the LiteLLM repository.

## OVERVIEW

LiteLLM is a unified interface for 100+ LLMs that:
- Translates inputs to provider-specific completion, embedding, and image generation endpoints
- Provides consistent OpenAI-format output across all providers
- Includes retry/fallback logic across multiple deployments (Router)
- Offers a proxy server (LLM Gateway) with budgets, rate limits, and authentication
- Supports advanced features like function calling, streaming, caching, and observability

## REPOSITORY STRUCTURE

### Core Components
- `litellm/` - Main library code
  - `llms/` - Provider-specific implementations (OpenAI, Anthropic, Azure, etc.)
  - `proxy/` - Proxy server implementation (LLM Gateway)
  - `router_utils/` - Load balancing and fallback logic
  - `types/` - Type definitions and schemas
  - `integrations/` - Third-party integrations (observability, caching, etc.)

### Key Directories
- `tests/` - Comprehensive test suites
- `docs/my-website/` - Documentation website
- `ui/litellm-dashboard/` - Admin dashboard UI
- `enterprise/` - Enterprise-specific features

## DEVELOPMENT GUIDELINES

### MAKING CODE CHANGES

1. **Provider Implementations**: When adding/modifying LLM providers:
   - Follow existing patterns in `litellm/llms/{provider}/`
   - Implement proper transformation classes that inherit from `BaseConfig`
   - Support both sync and async operations
   - Handle streaming responses appropriately
   - Include proper error handling with provider-specific exceptions

2. **Type Safety**: 
   - Use proper type hints throughout
   - Update type definitions in `litellm/types/`
   - Ensure compatibility with both Pydantic v1 and v2

3. **Testing**:
   - Add tests in appropriate `tests/` subdirectories
   - Include both unit tests and integration tests
   - Test provider-specific functionality thoroughly
   - Consider adding load tests for performance-critical changes

### MAKING CODE CHANGES FOR THE UI (IGNORE FOR BACKEND)

1. **Tremor is DEPRECATED, do not use Tremor components in new features/changes**
   - The only exception is the Tremor Table component and its required Tremor Table sub components.

2. **Use Common Components as much as possible**:
   - These are usually defined in the `common_components` directory
   - Use these components as much as possible and avoid building new components unless needed

3. **Testing**:
   - The codebase uses **Vitest** and **React Testing Library**
   - **Query Priority Order**: Use query methods in this order: `getByRole`, `getByLabelText`, `getByPlaceholderText`, `getByText`, `getByTestId`
   - **Always use `screen`** instead of destructuring from `render()` (e.g., use `screen.getByText()` not `getByText`)
   - **Wrap user interactions in `act()`**: Always wrap `fireEvent` calls with `act()` to ensure React state updates are properly handled
   - **Use `query` methods for absence checks**: Use `queryBy*` methods (not `getBy*`) when expecting an element to NOT be present
   - **Test names must start with "should"**: All test names should follow the pattern `it("should ...")`
   - **Mock external dependencies**: Check `setupTests.ts` for global mocks and mock child components/networking calls as needed
   - **Structure tests properly**:
     - First test should verify the component renders successfully
     - Subsequent tests should focus on functionality and user interactions
     - Use `waitFor` for async operations that aren't already awaited
   - **Avoid using `querySelector`**: Prefer React Testing Library queries over direct DOM manipulation

### IMPORTANT PATTERNS

1. **Function/Tool Calling**:
   - LiteLLM standardizes tool calling across providers
   - OpenAI format is the standard, with transformations for other providers
   - See `litellm/llms/anthropic/chat/transformation.py` for complex tool handling

2. **Streaming**:
   - All providers should support streaming where possible
   - Use consistent chunk formatting across providers
   - Handle both sync and async streaming

3. **Error Handling**:
   - Use provider-specific exception classes
   - Maintain consistent error formats across providers
   - Include proper retry logic and fallback mechanisms

4. **Configuration**:
   - Support both environment variables and programmatic configuration
   - Use `BaseConfig` classes for provider configurations
   - Allow dynamic parameter passing

## PROXY SERVER (LLM GATEWAY)

The proxy server is a critical component that provides:
- Authentication and authorization
- Rate limiting and budget management
- Load balancing across multiple models/deployments
- Observability and logging
- Admin dashboard UI
- Enterprise features

Key files:
- `litellm/proxy/proxy_server.py` - Main server implementation
- `litellm/proxy/auth/` - Authentication logic
- `litellm/proxy/management_endpoints/` - Admin API endpoints

**Database (proxy)**: Use Prisma model methods (`prisma_client.db.<model>.upsert`, `.find_many`, `.find_unique`, etc.), not raw SQL (`execute_raw`/`query_raw`). See COMMON PITFALLS for details.

## MCP (MODEL CONTEXT PROTOCOL) SUPPORT

LiteLLM supports MCP for agent workflows:
- MCP server integration for tool calling
- Transformation between OpenAI and MCP tool formats
- Support for external MCP servers (Zapier, Jira, Linear, etc.)
- See `litellm/experimental_mcp_client/` and `litellm/proxy/_experimental/mcp_server/`

## RUNNING SCRIPTS

Prefer the repo-local virtualenv for commands in this repository.

- Use `.venv/bin/python script.py` to run Python scripts in the project environment.
- Use `.venv/bin/pytest ...` for tests.
- Use `.venv/bin/python -m pytest ...` if `pytest` is not directly available.
- Only fall back to `poetry run ...` if the repo-local `.venv` is unavailable.

## GITHUB TEMPLATES

When opening issues or pull requests, follow these templates:

### Bug Reports (`.github/ISSUE_TEMPLATE/bug_report.yml`)
- Describe what happened vs. expected behavior
- Include relevant log output
- Specify LiteLLM version
- Indicate if you're part of an ML Ops team (helps with prioritization)

### Feature Requests (`.github/ISSUE_TEMPLATE/feature_request.yml`)
- Clearly describe the feature
- Explain motivation and use case with concrete examples

### Pull Requests (`.github/pull_request_template.md`)
- Add at least 1 test in `tests/litellm/`
- Ensure `make test-unit` passes


## TESTING CONSIDERATIONS

1. **Provider Tests**: Test against real provider APIs when possible
2. **Proxy Tests**: Include authentication, rate limiting, and routing tests
3. **Performance Tests**: Load testing for high-throughput scenarios
4. **Integration Tests**: End-to-end workflows including tool calling

## DOCUMENTATION

- Keep documentation in sync with code changes
- Update provider documentation when adding new providers
- Include code examples for new features
- Update changelog and release notes

## SECURITY CONSIDERATIONS

- Handle API keys securely
- Validate all inputs, especially for proxy endpoints
- Consider rate limiting and abuse prevention
- Follow security best practices for authentication

### Upstream security advisory sync status

As of 2026-07-17, the following GitHub Security Advisories have already been
synced into this branch. During future upstream sync/rebase work, use this list
to quickly filter advisory-related upstream commits or release notes that are
already covered here:

- `GHSA-wxxx-gvqv-xp7p` — Sandbox escape in custom-code guardrail.
  - Advisory: https://github.com/advisories/GHSA-wxxx-gvqv-xp7p
  - Upstream fixed in `1.83.11` by replacing the hand-rolled custom-code
    guardrail sandbox with `RestrictedPython`.
- `GHSA-4xpc-pv4p-pm3w` — Authentication bypass via Host Header injection.
  - Advisory: https://github.com/BerriAI/litellm/security/advisories/GHSA-4xpc-pv4p-pm3w
  - Upstream fixed in `1.84.0` with follow-up path-handling hardening backported
    to maintained release lines; route/auth decisions should use ASGI scope path
    data rather than Host-reconstructed URL paths.
- `GHSA-q775-qw9r-2r4g` — `/key/generate` delegated-budget ceiling bypass.
  - Public follow-up reference: https://github.com/BerriAI/litellm/issues/29073
  - Upstream fix introduced a caller-budget ceiling for key generation. When
    syncing newer upstream releases, watch for follow-up fixes around UI/SSO/CLI
    session-token budget-ceiling handling and preserve the secure behavior.
- `GHSA-f9v2-4w9p-2cwc` / `CVE-2026-34182` — critical OpenSSL vulnerability.
  - The Wolfi build and runtime images in `Dockerfile` and `docker/Dockerfile*`
    use the fixed upstream stable digest from `fda08dd727`. After changing the
    base image, verify the installed `openssl` and `libssl3` packages in the
    built runtime image remain at the fixed versions.
- `GHSA-4g5m-c9r5-49xf` / `CVE-2026-59819` — server secret disclosure through
  request-supplied environment and OIDC file references.
  - Backported from `06a0d4498a`: `/health/test_connection` rejects nested
    `os.environ/` request values, dynamic callback request metadata does not
    resolve environment references, and `oidc/file/` reads are restricted to
    configured credential directories.
  - Preserve the related cross-team deployment ownership authorization from
    `a2f5bb1868`: health probes that load a deployment must authorize against
    the loaded deployment's `model_info`, not caller-supplied ownership data.
- SpendLogs Bearer key hardening from `b487a80f4c`.
  - `litellm/proxy/spend_tracking/spend_tracking_utils.py` strips the Bearer
    scheme and hashes `sk-` keys in both the SpendLogs API-key column and
    `metadata.user_api_key` fallback paths.
- `GHSA-7488-6r32-c95q` / `CVE-2026-59822` — MCP authentication bypass.
  - Backported from `73869f0faf`: only exact `/.well-known/` paths are public,
    and a failed LiteLLM Bearer authentication may fall back to OAuth2
    passthrough only when every explicitly resolved target MCP server is
    operator-configured with `auth_type=oauth2`; unresolved, empty, mixed, and
    non-OAuth2 targets fail closed.
- RCEliteLLM chain (LiteLLM ≤ 1.83.14 RCE via master-key leak + Jinja2 SSTI).
  - Upstream fixed in `1.84.0-rc.1` / commits:
    - `f2f1e3a0ba` / `22c01adeb2` / `1ebb192cbe` (PR #26851): remove
      `get_secret()` from key/team logging callback metadata conversion, reject
      `os.environ/` callback refs via `validate_no_callback_env_reference`, and
      ignore invalid callback metadata rows.
    - `15d4d51453`: when a custom Langsmith/Langfuse host/base_url is supplied
      dynamically, do not fall back to process environment credentials
      (`allow_env_credentials=False`).
  - SSTI stage was already covered here via `ImmutableSandboxedEnvironment` in
    GitLab/BitBucket/Dotprompt/Arize prompt managers.
  - Local review on 2026-07-17 found the backport incomplete: the Langfuse
    prompt-management dynamic-host calls still permit environment credential
    fallback, and `/langfuse/*` passthrough still lacks upstream credential
    isolation and SSRF/path hardening. Do not mark this advisory fully fixed.
  - This branch disables all proxy-side Langfuse, Langfuse OTEL, and Langsmith
    features as a compensating control. Do not re-enable any of them until the
    omitted parts of upstream `15d4d51453` and their security tests are
    backported and reviewed.

Do not treat this section as a local patch inventory. It is only a security
advisory filter for future upstream syncs. If a future upstream sync contains
newer commits for one of the advisories above, compare behavior and regression
tests before dropping local coverage.

### Local proxy authorization invariants

The following controls are intentional security boundaries. Preserve them
during upstream syncs and when adding new key-management routes:

- Non-`proxy_admin` callers must not set virtual-key `metadata`, including
  explicit `null` or empty mappings. Presence is security-significant because
  update/regenerate paths can use empty values to clear an existing policy.
  This applies to every key write path, including `/key/generate`,
  `/key/service-account/generate`, `/key/update`, `/key/regenerate`, and bulk
  or indirect helpers. Only omitted metadata is valid.
- Non-`proxy_admin` callers must not set the key control fields
  `allowed_routes`, `allowed_passthrough_routes`, `config`, `aliases`,
  `router_settings`, `access_group_ids`, `permissions`, `object_permission`,
  `tags`, `guardrails`, `policies`, `prompts`, or `blocked`, and must not mint
  `key_type=management`. These fields can affect route/model authorization,
  fallbacks, delegated object access, policy enforcement, or management access.
- Authentication builders may return through many JWT, OAuth2, master-key,
  pass-through, custom-header, and DB-fallback paths. All non-exempt paths must
  converge on `_run_centralized_common_checks`; never add a wrapper return that
  bypasses it. DB-unavailable fallback identities must remain restricted
  internal users, never proxy admins.
- `metadata.allowed_passthrough_routes` is never a general route grant. The
  RBAC fallback may honor it only after confirming that the requested path is
  an operator-registered passthrough endpoint. Keep this runtime check even if
  management endpoints also reject new non-admin values, because existing DB
  rows and future write paths must fail closed.
- Do not relax these restrictions to a reserved-key denylist without tracing
  every metadata consumer. Key metadata has historically been interpreted as
  callback, guardrail, rate-limit, and route-control configuration.
- Before changing these controls, run at least:

  ```bash
  .venv/bin/pytest tests/test_litellm/proxy/auth/test_route_checks.py -q
  .venv/bin/pytest tests/test_litellm/proxy/management_endpoints/test_key_management_endpoints.py -q
  ```

  Also verify end to end that a normal internal user can create a key, set a
  budget within its delegation ceiling, and use the key, while non-empty
  metadata and the restricted control fields receive HTTP 403.

## ENTERPRISE FEATURES

- Some features are enterprise-only
- Check `enterprise/` directory for enterprise-specific code
- Maintain compatibility between open-source and enterprise versions

## COMMON PITFALLS TO AVOID

1. **Breaking Changes**: LiteLLM has many users - avoid breaking existing APIs
2. **Provider Specifics**: Each provider has unique quirks - handle them properly
3. **Rate Limits**: Respect provider rate limits in tests
4. **Memory Usage**: Be mindful of memory usage in streaming scenarios
5. **Dependencies**: Keep dependencies minimal and well-justified
6. **UI/Backend Contract Mismatch**: When adding a new entity type to the UI, always check whether the backend endpoint accepts a single value or an array. Match the UI control accordingly (single-select vs. multi-select) to avoid silently dropping user selections
7. **Missing Tests for New Entity Types**: When adding a new entity type (e.g., in `EntityUsage`, `UsageViewSelect`), always add corresponding tests in the existing test files and update any icon/component mocks
8. **Raw SQL in proxy DB code**: Do not use `execute_raw` or `query_raw` for proxy database access. Use Prisma model methods (e.g. `prisma_client.db.litellm_tooltable.upsert()`, `.find_many()`, `.find_unique()`) so behavior stays consistent with the schema, the client stays mockable in tests, and you avoid the pitfalls of hand-written SQL (parameter ordering, type casting, schema drift)

8. **Do not hardcode model-specific flags**: Put model-specific capability flags in `model_prices_and_context_window.json` and read them via `get_model_info` (or existing helpers like `supports_reasoning`). This prevents users from needing to upgrade LiteLLM each time a new model supports a feature.

   **Example of BAD** (hardcoded model checks):

   ```python
   @staticmethod
   def _is_effort_supported_model(model: str) -> bool:
       """Check if the model supports the output_config.effort parameter..."""
       model_lower = model.lower()
       if AnthropicConfig._is_claude_4_6_model(model):
           return True
       return any(
           v in model_lower for v in ("opus-4-5", "opus_4_5", "opus-4.5", "opus_4.5")
       )
   ```

   **Example of GOOD** (config-driven or helper that reads from config):

   ```python
   if (
       "claude-3-7-sonnet" in model
       or AnthropicConfig._is_claude_4_6_model(model)
       or supports_reasoning(
           model=model,
           custom_llm_provider=self.custom_llm_provider,
       )
   ):
       ...
   ```

   Using helpers like `supports_reasoning` (which read from `model_prices_and_context_window.json` / `get_model_info`) allows future model updates to "just work" without code changes.

9. **Never close HTTP/SDK clients on cache eviction**: Do not add `close()`, `aclose()`, or `create_task(close_fn())` inside `LLMClientCache._remove_key()` or any cache eviction path. Evicted clients may still be held by in-flight requests; closing them causes `RuntimeError: Cannot send a request, as the client has been closed.` in production after the cache TTL (1 hour) expires. Connection cleanup is handled at shutdown by `close_litellm_async_clients()`. See PR #22247 for the full incident history.

## HELPFUL RESOURCES

- Main documentation: https://docs.litellm.ai/
- Provider-specific docs in `docs/my-website/docs/providers/`
- Admin UI for testing proxy features

## WHEN IN DOUBT

- Follow existing patterns in the codebase
- Check similar provider implementations
- Ensure comprehensive test coverage
- Update documentation appropriately
- Consider backward compatibility impact

## Cursor Cloud specific instructions

### Environment

- Poetry is installed in `~/.local/bin`; the update script ensures it is on `PATH`.
- Python 3.12, Node 22 are pre-installed.
- The virtual environment lives under `~/.cache/pypoetry/virtualenvs/`.

### Running the proxy server

Start the proxy with a config file:

```bash
poetry run litellm --config dev_config.yaml --port 4000
```

The proxy takes ~15-20 seconds to fully start (it runs Prisma migrations on boot). Wait for `/health` to return before sending requests. Without a PostgreSQL `DATABASE_URL`, the proxy connects to a default Neon dev database embedded in the `litellm-proxy-extras` package.

### Running tests

See `CLAUDE.md` and the `Makefile` for standard commands. Key notes:

- `psycopg-binary` must be installed (`.venv/bin/pip install psycopg-binary`) because the pytest-postgresql plugin requires it and the lock file only includes `psycopg` (no binary).
- `openapi-core` must be installed (`.venv/bin/pip install openapi-core`) for the OpenAPI compliance tests in `tests/test_litellm/interactions/`.
- The `--timeout` pytest flag is NOT available; don't pass it.
- Unit tests: `.venv/bin/pytest tests/test_litellm/ -x -vv -n 4`
- Black `--check` may report pre-existing formatting issues; this does not block test runs.
- If `poetry install` fails with "pyproject.toml changed significantly since poetry.lock was last generated", run `poetry lock` first to regenerate the lock file.

### Commit, image build, and local blue/green deployment

When asked to ship the current repo changes to the local LiteLLM deployment:

1. Inspect the worktree first:

   ```bash
   git status --short --branch
   git diff --stat
   git diff
   ```

2. Run the most relevant targeted tests before committing. For route-check changes, for example:

   ```bash
   .venv/bin/pytest tests/proxy_admin_ui_tests/test_route_check_unit_tests.py -q
   ```

3. Commit and push the current branch:

   ```bash
   git add <changed files>
   git commit -m "<descriptive message>"
   git push
   ```

4. Build the local Docker image from the repo root after the commit is created. Use an immutable commit-specific tag by default: replace `<commit>` with the real short commit SHA from `git rev-parse --short=12 HEAD`. Only use the literal tag `litellm:commit` if the user explicitly asks for that exact tag.

   ```bash
   COMMIT_TAG="$(git rev-parse --short=12 HEAD)"
   docker build -t "litellm:${COMMIT_TAG}" .
   docker image ls "litellm:${COMMIT_TAG}"
   ```

5. Use the deployment script in `~/litellm` for traffic switching. Do **not** manually stop LiteLLM containers with `docker stop`; let the deployment script manage blue/green cutover, drain, and old-service shutdown.

   ```bash
   cd ~/litellm
   ./deploy-litellm.sh status
   ./deploy-litellm.sh switch --image "litellm:${COMMIT_TAG}"
   ./deploy-litellm.sh status
   ```

   Notes:
   - `deploy-litellm.sh` does not build images; pass the exact image tag built in the previous step after `--image`.
   - The script starts the inactive `litellm_blue`/`litellm_green` service, waits for `/health/readiness`, rewrites Traefik `ai-svc` to the new backend port, then drains established TCP connections on the old backend before stopping it.
   - Drain defaults are `DRAIN_TIMEOUT=120` seconds and `DRAIN_IDLE_SECONDS=5` continuous seconds with zero established connections. `DRAIN_SECONDS` is retained as a backward-compatible default for `DRAIN_TIMEOUT`; `DRAIN_TIMEOUT=0` waits indefinitely.
   - Only set `STOP_OLD=0` if explicitly asked to keep the old backend running.

6. Verify the active backend and readiness after the switch:

   ```bash
   cd ~/litellm
   ./deploy-litellm.sh status
   curl -fsS http://127.0.0.1:<active-port>/health/readiness
   ```

### Container hotpatching

When using `docker cp` to hotpatch LiteLLM Python files in the running containers, copy the file to both code locations inside each container:

- `/app/...`
- `/usr/lib/python3.13/site-packages/...`

For the local LiteLLM containers used here, there may be two app containers to patch, for example:

- `litellm-litellm-1`
- `litellm-litellm-test-1`

Do not assume patching only one path or one container is enough.

### Lint

```bash
cd litellm && poetry run ruff check .
```

Ruff is the primary fast linter. For the full lint suite (including mypy, black, circular imports), run `make lint` per `CLAUDE.md`.

### UI Dashboard development

- The UI is at `ui/litellm-dashboard/`. Run `npm run dev` from that directory for the Next.js dev server on port 3000.
- The proxy at port 4000 serves a **pre-built** static UI from `litellm/proxy/_experimental/out/`. After making UI code changes, you must run `npm run build` in the dashboard directory and copy the output: `cp -r ui/litellm-dashboard/out/* litellm/proxy/_experimental/out/` for the proxy to serve the updated UI.
- SVGs used as provider logos (loaded via `<img>` tags) must NOT use `fill="currentColor"` — replace with an explicit color like `#000000` or use the `-color` variant from lobehub icons, since CSS color inheritance does not work inside `<img>` elements.
- Provider logos live in `ui/litellm-dashboard/public/assets/logos/` (source) and `litellm/proxy/_experimental/out/assets/logos/` (pre-built). Both locations must have the file for it to work in dev and proxy-served modes.
- UI Vitest tests: `cd ui/litellm-dashboard && npx vitest run`
