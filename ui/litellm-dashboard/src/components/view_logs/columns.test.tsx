import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createColumns, type LogEntry } from "./columns";

vi.mock("antd", () => ({
  Tooltip: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div data-testid="tooltip" data-title={title}>
      {children}
    </div>
  ),
}));

const renderKeyNameCell = (row: Pick<LogEntry, "metadata" | "user_api_key_user_name">) => {
  const keyNameColumn = createColumns().find((column) => column.header === "Key Name");
  if (!keyNameColumn?.cell) throw new Error("Key Name column not found");

  const Cell = keyNameColumn.cell as (info: any) => React.ReactNode;
  render(<>{Cell({ getValue: () => row.metadata?.user_api_key_alias, row: { original: row } })}</>);
};

describe("view logs columns", () => {
  it("should render the key name", () => {
    renderKeyNameCell({ metadata: { user_api_key_alias: "production-key" } });

    expect(screen.getByText("production-key")).toBeInTheDocument();
  });

  it("should show the resolved key owner name in the key name tooltip", () => {
    renderKeyNameCell({
      metadata: {
        user_api_key_alias: "production-key",
        user_api_key_user_id: "user-123",
      },
      user_api_key_user_name: "Alice",
    });

    expect(screen.getByTestId("tooltip")).toHaveAttribute("data-title", "Alice");
  });

  it("should fall back to the key owner user ID when no resolved name is available", () => {
    renderKeyNameCell({
      metadata: {
        user_api_key_alias: "production-key",
        user_api_key_user_id: "user-123",
      },
    });

    expect(screen.getByTestId("tooltip")).toHaveAttribute("data-title", "user-123");
  });
});
