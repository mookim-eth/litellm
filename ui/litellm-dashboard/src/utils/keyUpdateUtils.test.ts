import { describe, expect, it } from "vitest";
import { mapEmptyStringToNull, sanitizeNonAdminKeyPayload } from "./keyUpdateUtils";

describe("keyUpdateUtils", () => {
  it("should map empty string to null", () => {
    expect(mapEmptyStringToNull("")).toBeNull();
  });

  it("should return the original string otherwise", () => {
    expect(mapEmptyStringToNull("500")).toBe("500");
  });

  it("should omit policy fields from non-admin key payloads", () => {
    const payload = sanitizeNonAdminKeyPayload({
      key: "key-1",
      key_alias: "updated",
      max_budget: 10,
      models: ["gpt-4"],
      metadata: {},
      tags: [],
      policies: [],
      tpm_limit: null,
      rpm_limit: null,
      budget_duration: null,
      object_permission: {},
      vector_stores: [],
      logging_settings: [],
    });

    expect(payload).toEqual({
      key: "key-1",
      key_alias: "updated",
      max_budget: 10,
      models: ["gpt-4"],
    });
  });
});
