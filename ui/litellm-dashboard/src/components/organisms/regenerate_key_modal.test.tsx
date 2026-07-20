import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { KeyResponse } from "../key_team_helpers/key_list";
import { regenerateKeyCall } from "../networking";
import { RegenerateKeyModal } from "./regenerate_key_modal";

const { formState, formMock } = vi.hoisted(() => {
  const formState = { current: {} as Record<string, unknown> };
  const formMock = {
    setFieldsValue: vi.fn((values: Record<string, unknown>) => {
      Object.assign(formState.current, values);
    }),
    validateFields: vi.fn(async () => ({ ...formState.current })),
    resetFields: vi.fn(() => {
      formState.current = {};
    }),
  };
  return { formState, formMock };
});

let authorizedState = {
  accessToken: "access-token",
  userRole: "Internal User",
};

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => authorizedState,
}));

vi.mock("../networking", () => ({
  regenerateKeyCall: vi.fn().mockResolvedValue({ key: "regenerated-key" }),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

vi.mock("react-copy-to-clipboard", () => ({
  CopyToClipboard: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("@tremor/react", () => {
  const Stub = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return {
    Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button {...props}>{children}</button>
    ),
    Col: Stub,
    Grid: Stub,
    Text: Stub,
    TextInput: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
    Title: Stub,
  };
});

vi.mock("antd", () => {
  const React = require("react");
  const Form = ({ children }: { children?: React.ReactNode }) => <form>{children}</form>;
  Form.Item = ({ children, label, name }: { children?: React.ReactNode; label?: string; name?: string }) => (
    <label>
      {label}
      {name && React.isValidElement(children)
        ? React.cloneElement(children as React.ReactElement<Record<string, unknown>>, {
            value: formState.current[name] ?? "",
            onChange: (event: React.ChangeEvent<HTMLInputElement>) => {
              formState.current[name] = event.target.value;
            },
          })
        : children}
    </label>
  );
  Form.useForm = () => [formMock];

  return {
    Form,
    InputNumber: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input type="number" {...props} />,
    Modal: ({ children, footer, open }: { children?: React.ReactNode; footer?: React.ReactNode; open?: boolean }) =>
      open ? (
        <div>
          {children}
          {footer}
        </div>
      ) : null,
  };
});

const selectedToken = {
  token: "token-id",
  key_alias: "example-key",
  max_budget: 10,
  tpm_limit: 100,
  rpm_limit: 20,
  duration: "30d",
} as KeyResponse;

describe("RegenerateKeyModal", () => {
  beforeEach(() => {
    authorizedState = {
      accessToken: "access-token",
      userRole: "Internal User",
    };
    formState.current = {};
    vi.clearAllMocks();
  });

  it("should render successfully", () => {
    render(<RegenerateKeyModal selectedToken={selectedToken} visible onClose={vi.fn()} />);

    expect(screen.getByText("Key Alias")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Regenerate" })).toBeInTheDocument();
  });

  it("should omit proxy-admin controls when a non-admin regenerates a key", async () => {
    render(<RegenerateKeyModal selectedToken={selectedToken} visible onClose={vi.fn()} />);

    expect(screen.queryByText("TPM Limit")).not.toBeInTheDocument();
    expect(screen.queryByText("RPM Limit")).not.toBeInTheDocument();
    expect(screen.queryByText("Grace Period (eg: 24h, 2d)")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));

    await waitFor(() => expect(regenerateKeyCall).toHaveBeenCalledOnce());
    const request = vi.mocked(regenerateKeyCall).mock.calls[0][2];
    expect(request).not.toHaveProperty("tpm_limit");
    expect(request).not.toHaveProperty("rpm_limit");
    expect(request).not.toHaveProperty("grace_period");
  });

  it("should allow a proxy admin to request a grace period", async () => {
    authorizedState = {
      accessToken: "access-token",
      userRole: "Admin",
    };
    render(<RegenerateKeyModal selectedToken={selectedToken} visible onClose={vi.fn()} />);

    expect(screen.getByPlaceholderText("e.g. 24h, 2d (empty = immediate revoke)")).toBeInTheDocument();
    formState.current.grace_period = "24h";
    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));

    await waitFor(() => expect(regenerateKeyCall).toHaveBeenCalledOnce());
    expect(vi.mocked(regenerateKeyCall).mock.calls[0][2]).toMatchObject({
      grace_period: "24h",
      tpm_limit: 100,
      rpm_limit: 20,
    });
  });
});
