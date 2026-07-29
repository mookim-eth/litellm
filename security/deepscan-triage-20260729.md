# Deep security scan triage — 2026-07-29

Scope: candidates from `/root/workspace/codex-security-runs/litellm-deep-20260729T034837Z-custom-embedded` against the current local production deployment image `litellm:6019ca0bfe9d` and `/root/litellm/config.yaml`. Production was inspected read-only; no production container/config changes were made.

## Patched in this branch

1. `azure_ai_ocr_url_fetch_ssrf` — Not enabled by current production model config, but the shared image URL fetch helper was hardened with SSRF-safe URL validation because the vulnerable fetch path is generic.
2. `dotprompt_converter_filename_traversal` — Authenticated utility route is mounted; fixed uploaded `.prompt` filenames to reject absolute paths and path separators, and use a confined temporary directory.
5. `unauth_operational_telemetry_exposure` — Current production exposes the proxy publicly and has telemetry/debug routes mounted; fixed `/provider/budgets`, `/debug/asyncio-tasks`, `/otel-spans`, and profiled `/memory-usage` to require proxy admin/viewer auth.
16. `rag_query_vector_store_auth_bypass` — RAG routes are mounted; fixed centralized vector-store access extraction to include `retrieval_config.vector_store_id`.
17. `rag_ingest_file_url_ssrf` — RAG routes are mounted; fixed file URL ingestion to use SSRF-safe redirect-aware fetch.
18. `global-spend-reset-authz-bypass` — Current production uses the DB-backed proxy; fixed `/global/spend/reset` to require proxy admin.
19. `team-filter-ui-cross-team-disclosure` — Current production uses the DB-backed proxy; fixed `/team/filter/ui` to require proxy admin/viewer.
20. `user-filter-ui-directory-disclosure` — Current production uses the DB-backed proxy and default settings; fixed `/user/filter/ui` so non-admin callers are restricted to self when org-scoped search is not enabled.
21. `global-spend-analytics-cross-tenant-disclosure` — Current production uses spend logging; fixed global spend/tag/end-user analytics endpoints to require proxy admin/viewer, preserving existing scoped internal-user behavior for endpoints that already had it.
11. `vector_store_file_managed_object_authz_bypass` — Vector-store file endpoints are mounted; fixed managed vector-store file operations to enforce object/team access before provider dispatch.
23. `vector_store_update_cross_team_takeover` — Vector-store endpoints are mounted; fixed `/vector_store/update` to perform the same object access check used by info/delete before mutation.

## Skipped / not enough current-production impact

3. `litellm_proxy_skills_cross_tenant_access` — No evidence current production enables/uses DB-backed LiteLLM-managed skills. Needs a separate feature-specific review if skills are enabled.
4. `mcp_byok_oauth_user_credential_injection` — No MCP/BYOK configuration was found in current production environment/config. Skipped for current production impact.
6. `cli_sso_attacker_chosen_poll_key_account_takeover` — Current production container environment does not configure Google/Microsoft/generic SSO. Skipped for current production impact.
7. `cli_sso_existing_key_oauth_state_leak` — Same SSO precondition as above; skipped for current production impact.
8. `mcp_oauth_public_token_broker` — No MCP OAuth server configuration found in current production environment/config. Skipped for current production impact.
9. `delegated_expiry_ceiling_generate_bypass` — Already covered by the branch's delegated virtual-key authority hardening noted in `AGENTS.md`; skipped as not newly actionable here.
10. `org_admin_user_new_global_key_scope_bypass` — Already covered by organization-admin boundary fixes noted in `AGENTS.md`; skipped as not newly actionable here.
12. `responses_polling_id_cross_user_access` — Requires background Responses polling ID disclosure; not prioritized over current directly reachable DB/telemetry issues in this pass.
13. `mcp_access_groups_authenticated_info_leak` — MCP feature not configured in current production; skipped for current production impact.
14. `sso_debug_callback_inline_json_xss` — SSO not configured in current production; skipped for current production impact.
15. `fine_tuning_provider_job_authz_bypass` — No fine-tuning provider configuration found in current production config; skipped for current production impact.
22. `vector_store_auto_embedding_key_leak` — Current production config only showed ChatGPT model entries without embedding-provider API-key config; skipped for current production impact. If embedding deployments are added to DB/config, revisit.
24. `disable_admin_endpoints_route_classifier_bypass` — Current production does not set `DISABLE_ADMIN_ENDPOINTS=true`; skipped for current production impact.
