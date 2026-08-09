/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  credentialListCall,
  getCallbacksCall,
  getPassThroughEndpointsCall,
  latestHealthChecksCall,
} from "@/components/networking";
import ModelsAndEndpointsView from "./ModelsAndEndpointsView";

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();
Object.defineProperty(window, "localStorage", { value: localStorageMock });

// Minimal stubs to avoid Next.js router and network usage during render
vi.mock("@/components/networking", () => ({
  credentialListCall: vi.fn().mockResolvedValue({ credentials: [] }),
  modelInfoCall: vi.fn().mockResolvedValue({ data: [] }),
  modelCostMap: vi.fn().mockResolvedValue({}),
  getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: {} }),
  getCallbacksCall: vi.fn().mockResolvedValue({ router_settings: {} }),
  setCallbacksCall: vi.fn().mockResolvedValue(undefined),
  getUiSettings: vi.fn().mockResolvedValue({ values: {} }),
  latestHealthChecksCall: vi.fn().mockResolvedValue({ latest_health_checks: {} }),
  getModelCostMapReloadStatus: vi.fn().mockResolvedValue({}),
  getModelCostMapSource: vi.fn().mockResolvedValue({
    source: "local",
    url: null,
    is_env_forced: true,
    fallback_reason: null,
    model_count: 0,
  }),
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/components/ModelAnalyticsTab/ModelAnalyticsTab", () => ({
  default: () => null,
}));

vi.mock("@/components/add_model/add_auto_router_tab", () => ({
  default: () => null,
}));

vi.mock("@/components/add_model/AddModelForm", () => ({
  default: () => null,
}));

vi.mock("@/components/add_model/add_model_tab", () => ({
  default: () => <div>Add model panel content</div>,
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/components/AllModelsTab", async () => {
  const { TabPanel } = await import("@tremor/react");
  return {
    default: () => (
      <TabPanel>
        <div>All models panel content</div>
      </TabPanel>
    ),
  };
});

const mockCredentialsPanel = vi.fn(() => <div>Credentials panel content</div>);
vi.mock("@/components/model_add/credentials", () => ({
  default: () => mockCredentialsPanel(),
}));

const mockPassThroughSettings = vi.fn(() => <div>Pass-through panel content</div>);
vi.mock("@/components/pass_through_settings", () => ({
  default: () => mockPassThroughSettings(),
}));

const mockHealthCheckComponent = vi.fn((_props: { all_models_on_proxy?: string[] }) => null);
vi.mock("@/components/model_dashboard/HealthCheckComponent", () => ({
  default: (props: { all_models_on_proxy?: string[] }) => {
    mockHealthCheckComponent(props);
    return <div>Health status panel content</div>;
  },
}));

const mockModelRetrySettingsTab = vi.fn(() => <div>Retry settings panel content</div>);
vi.mock("@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab", async () => {
  const { TabPanel } = await import("@tremor/react");
  return {
    default: () => <TabPanel>{mockModelRetrySettingsTab()}</TabPanel>,
  };
});

const mockModelGroupAliasSettings = vi.fn(() => <div>Model alias panel content</div>);
vi.mock("@/components/model_group_alias_settings", () => ({
  default: () => mockModelGroupAliasSettings(),
}));

const mockPriceDataManagementTab = vi.fn(() => <div>Price data panel content</div>);
vi.mock("@/app/(dashboard)/models-and-endpoints/components/PriceDataManagementTab", async () => {
  const { TabPanel } = await import("@tremor/react");
  return {
    default: () => <TabPanel>{mockPriceDataManagementTab()}</TabPanel>,
  };
});

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: () => ({
    teams: [],
    setTeams: vi.fn(),
  }),
}));

const mockUseModelsInfo = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: () => mockUseModelsInfo(),
}));

const mockUseUISettings = vi.fn();
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => mockUseUISettings(),
}));

const mockUseModelCostMap = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => mockUseModelCostMap(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

describe("ModelsAndEndpointsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseModelsInfo.mockReturnValue({
      data: { data: [] },
      isLoading: false,
      refetch: vi.fn(),
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
    });
    mockUseModelCostMap.mockReturnValue({
      data: {},
      isLoading: false,
      error: null,
    });
    mockUseAuthorized.mockReturnValue({
      accessToken: "123",
      token: "123",
      userRole: "Admin",
      userId: "123",
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("should render the models and endpoints view", async () => {
    const queryClient = createQueryClient();
    const { findByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );
    expect(await findByText("Model Management", {}, { timeout: 10000 })).toBeInTheDocument();
  });

  it("should show Missing provider banner by default", async () => {
    localStorageMock.clear();
    const queryClient = createQueryClient();
    const { findByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );
    expect(await findByText("Missing a provider?", {}, { timeout: 10000 })).toBeInTheDocument();
  });

  it("should hide Missing provider banner when dismiss button is clicked and persist to localStorage", async () => {
    localStorageMock.clear();
    const queryClient = createQueryClient();
    const { findByText, queryByText, container } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );

    // Wait for banner to appear
    expect(await findByText("Missing a provider?", {}, { timeout: 10000 })).toBeInTheDocument();

    // Find and click dismiss button (X button)
    const dismissButton = container.querySelector('button[aria-label="Dismiss banner"]');
    expect(dismissButton).not.toBeNull();
    fireEvent.click(dismissButton!);

    // Banner should be hidden
    expect(queryByText("Missing a provider?")).not.toBeInTheDocument();

    // LocalStorage should be updated
    expect(localStorageMock.getItem("hideMissingProviderBanner")).toBe("true");
  });

  it("should show compact Request Provider button when banner is dismissed", async () => {
    // Set localStorage to hide banner
    localStorageMock.setItem("hideMissingProviderBanner", "true");
    const queryClient = createQueryClient();
    const { findByText, queryByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );

    // Wait for component to render
    await findByText("Model Management", {}, { timeout: 10000 });

    // Banner should not be visible
    expect(queryByText("Missing a provider?")).not.toBeInTheDocument();

    // Compact Request Provider button should be visible in header
    const requestProviderLinks = document.querySelectorAll('a[href="https://models.litellm.ai/?request=true"]');
    // There should be a compact button when banner is hidden
    expect(requestProviderLinks.length).toBeGreaterThan(0);
  });

  it("should not mount admin-only panels or fetch admin-only settings for internal users", async () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: "123",
      token: "123",
      userRole: "Internal User",
      userId: "123",
    });

    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("Model Management", {}, { timeout: 10000 })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "LLM Credentials" })).not.toBeInTheDocument();
    expect(mockCredentialsPanel).not.toHaveBeenCalled();
    expect(mockPassThroughSettings).not.toHaveBeenCalled();
    expect(mockHealthCheckComponent).not.toHaveBeenCalled();
    expect(mockModelRetrySettingsTab).not.toHaveBeenCalled();
    expect(mockModelGroupAliasSettings).not.toHaveBeenCalled();
    expect(mockPriceDataManagementTab).not.toHaveBeenCalled();

    await waitFor(() => {
      expect(credentialListCall).not.toHaveBeenCalled();
      expect(getCallbacksCall).not.toHaveBeenCalled();
      expect(getPassThroughEndpointsCall).not.toHaveBeenCalled();
      expect(latestHealthChecksCall).not.toHaveBeenCalled();
    });
  });

  it("should pass model IDs (not model names) to HealthCheckComponent as all_models_on_proxy", async () => {
    mockHealthCheckComponent.mockClear();
    const modelDataWithIds = {
      data: [
        { model_name: "gpt-4", model_info: { id: "deployment-id-1" } },
        { model_name: "gpt-4", model_info: { id: "deployment-id-2" } },
      ],
    };
    mockUseModelsInfo.mockReturnValue({
      data: { data: modelDataWithIds.data },
      isLoading: false,
      refetch: vi.fn(),
    });

    const queryClient = createQueryClient();
    const { getByRole } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: modelDataWithIds.data }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );

    const healthStatusTab = getByRole("tab", { name: "Health Status" });
    await act(async () => {
      healthStatusTab.click();
    });

    expect(mockHealthCheckComponent).toHaveBeenCalled();
    const healthCheckProps = mockHealthCheckComponent.mock.calls[0][0];
    expect(healthCheckProps.all_models_on_proxy).toEqual(["deployment-id-1", "deployment-id-2"]);
    expect(healthCheckProps.all_models_on_proxy).not.toContain("gpt-4");
  });

  it("should show only the selected admin tab panel", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );

    const panels = [
      {
        tab: "LLM Credentials",
        content: await screen.findByText("Credentials panel content", {}, { timeout: 10000 }),
      },
      {
        tab: "Pass-Through Endpoints",
        content: screen.getByText("Pass-through panel content"),
      },
      {
        tab: "Health Status",
        content: screen.getByText("Health status panel content"),
      },
      {
        tab: "Model Retry Settings",
        content: screen.getByText("Retry settings panel content"),
      },
      {
        tab: "Model Group Alias",
        content: screen.getByText("Model alias panel content"),
      },
      {
        tab: "Price Data Reload",
        content: screen.getByText("Price data panel content"),
      },
    ];

    for (const selectedPanel of panels) {
      await act(async () => {
        fireEvent.click(screen.getByRole("tab", { name: selectedPanel.tab }));
      });

      for (const panel of panels) {
        const panelContainer = panel.content.closest("[aria-selected]");
        expect(panelContainer).not.toBeNull();
        expect(panelContainer).toHaveAttribute("aria-selected", panel === selectedPanel ? "true" : "false");
      }
    }
  });
});
