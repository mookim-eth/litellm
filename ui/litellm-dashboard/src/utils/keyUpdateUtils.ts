export function mapEmptyStringToNull(input: string): string | null {
  if (input === "") {
    return null;
  }

  return input;
}

// Keep aligned with _NON_ADMIN_RESTRICTED_KEY_CONTROL_FIELDS in
// key_management_endpoints.py. The extra UI-only fields below are transformed
// into metadata or object_permission before submission.
export const NON_ADMIN_RESTRICTED_KEY_FIELDS = [
  "agent_id",
  "allowed_routes",
  "allowed_passthrough_routes",
  "allowed_cache_controls",
  "allowed_vector_store_indexes",
  "config",
  "aliases",
  "router_settings",
  "access_group_ids",
  "permissions",
  "object_permission",
  "tags",
  "guardrails",
  "disable_global_guardrails",
  "policies",
  "prompts",
  "blocked",
  "budget_id",
  "budget_duration",
  "enforced_params",
  "grace_period",
  "max_parallel_requests",
  "model_max_budget",
  "model_rpm_limit",
  "model_tpm_limit",
  "project_id",
  "rpm_limit",
  "rpm_limit_type",
  "spend",
  "temp_budget_expiry",
  "temp_budget_increase",
  "tpm_limit",
  "tpm_limit_type",
  "metadata",
  "allowed_vector_store_ids",
  "allowed_mcp_servers_and_groups",
  "allowed_mcp_access_groups",
  "mcp_tool_permissions",
  "mcp_servers_and_groups",
  "vector_stores",
  "allowed_agents_and_groups",
  "agents_and_groups",
  "logging_settings",
  "disabled_callbacks",
] as const;

export function sanitizeNonAdminKeyPayload(payload: Record<string, any>): Record<string, any> {
  const sanitizedPayload = { ...payload };
  for (const field of NON_ADMIN_RESTRICTED_KEY_FIELDS) {
    delete sanitizedPayload[field];
  }
  return sanitizedPayload;
}
