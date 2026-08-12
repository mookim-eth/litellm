import { describe, expect, it } from "vitest";

import { ERROR_CODE_OPTIONS } from "./constants";

describe("ERROR_CODE_OPTIONS", () => {
  it("should include legacy HTTP 200 failure records", () => {
    expect(ERROR_CODE_OPTIONS).toContainEqual({
      label: "200 - Failed SSE Response (Legacy)",
      value: "200",
    });
  });
});
