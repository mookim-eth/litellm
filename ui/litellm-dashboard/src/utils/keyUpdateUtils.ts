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

function keyUpdateValuesEqual(left: any, right: any): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  if (left == null && right == null) {
    return true;
  }
  if (
    (right == null && Array.isArray(left) && left.length === 0) ||
    (left == null && Array.isArray(right) && right.length === 0)
  ) {
    return true;
  }
  if (
    (right == null && left && typeof left === "object" && Object.keys(left).length === 0) ||
    (left == null && right && typeof right === "object" && Object.keys(right).length === 0)
  ) {
    return true;
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => keyUpdateValuesEqual(value, right[index]))
    );
  }
  if (left && right && typeof left === "object" && typeof right === "object") {
    const leftKeys = Object.keys(left).sort();
    const rightKeys = Object.keys(right).sort();
    return (
      keyUpdateValuesEqual(leftKeys, rightKeys) &&
      leftKeys.every((key) => keyUpdateValuesEqual(left[key], right[key]))
    );
  }
  return false;
}

export function omitUnchangedKeyFields(
  payload: Record<string, any>,
  currentKey: Record<string, any>,
): Record<string, any> {
  const changedPayload: Record<string, any> = {};
  for (const [field, value] of Object.entries(payload)) {
    if (field === "key") {
      changedPayload[field] = value;
      continue;
    }
    if (field === "token" || value === undefined) {
      continue;
    }

    const currentValue = ["guardrails", "policies", "prompts"].includes(field)
      ? currentKey.metadata?.[field]
      : currentKey[field];
    if (!keyUpdateValuesEqual(value, currentValue)) {
      changedPayload[field] = value;
    }
  }
  return changedPayload;
}
