"use client";

import React, { Suspense, useEffect, useState } from "react";
import Navbar from "@/components/navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useRouter, useSearchParams } from "next/navigation";
import { DebugWarningBanner } from "@/components/DebugWarningBanner";

/** ---- BASE URL HELPERS ---- */
function normalizeBasePrefix(raw: string | undefined | null): string {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return "";
  const core = trimmed.replace(/^\/+/, "").replace(/\/+$/, "");
  return core ? `/${core}/` : "/";
}
const BASE_PREFIX = normalizeBasePrefix(process.env.NEXT_PUBLIC_BASE_URL);
function withBase(path: string): string {
  const body = path.startsWith("/") ? path.slice(1) : path;
  const combined = `${BASE_PREFIX}${body}`;
  return combined.startsWith("/") ? combined : `/${combined}`;
}
/** -------------------------------- */

/**
 * Pages that have been migrated to path-based routing under (dashboard)/.
 * When the leftnav triggers one of these, navigate to the path route instead
 * of the legacy query-param root page.
 *
 * Key = legacy page id used in leftnav, Value = route segment under (dashboard)/
 */
const MIGRATED_PAGES: Record<string, string> = {
  "api-reference": "api-reference",
};

function LayoutContent({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { accessToken, userRole, userId, userEmail, premiumUser } = useAuthorized();
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [isMobileViewport, setIsMobileViewport] = useState(false);
  const [page, setPage] = useState(() => {
    return searchParams.get("page") || "api-keys";
  });

  const handleSetPage = (newPage: string) => {
    // If the page has been migrated to path routing, navigate there
    const migratedRoute = MIGRATED_PAGES[newPage];
    if (migratedRoute) {
      router.push(withBase(migratedRoute));
      setPage(newPage);
      setMobileSidebarOpen(false);
      return;
    }

    // Otherwise, navigate back to the legacy root page with query params
    router.push(withBase(`?page=${newPage}`));
    setPage(newPage);
    setMobileSidebarOpen(false);
  };

  useEffect(() => {
    setPage(searchParams.get("page") || "api-keys");
  }, [searchParams]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia("(max-width: 767px)");
    const handleViewportChange = () => {
      const isMobile = mediaQuery.matches;
      setIsMobileViewport(isMobile);
      if (!isMobile) {
        setMobileSidebarOpen(false);
      }
    };

    handleViewportChange();
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", handleViewportChange);
      return () => mediaQuery.removeEventListener("change", handleViewportChange);
    }

    mediaQuery.addListener(handleViewportChange);
    return () => mediaQuery.removeListener(handleViewportChange);
  }, []);

  const toggleSidebar = () => {
    if (isMobileViewport) {
      setMobileSidebarOpen((v) => !v);
      return;
    }

    setSidebarCollapsed((v) => !v);
  };

  const closeMobileSidebar = () => setMobileSidebarOpen(false);

  return (
    <ThemeProvider accessToken={""}>
      <div className="flex flex-col min-h-screen">
        <Navbar
          isPublicPage={false}
          sidebarCollapsed={isMobileViewport ? !mobileSidebarOpen : sidebarCollapsed}
          onToggleSidebar={toggleSidebar}
          userID={userId}
          userEmail={userEmail}
          userRole={userRole}
          premiumUser={premiumUser}
          proxySettings={undefined}
          setProxySettings={() => {}}
          accessToken={accessToken}
          isDarkMode={false}
          toggleDarkMode={() => {}}
        />
        <DebugWarningBanner />
        <div className="flex flex-1 min-w-0 overflow-hidden">
          {mobileSidebarOpen && (
            <button
              aria-label="Close sidebar"
              className="fixed inset-0 top-14 z-20 bg-black/20 md:hidden"
              onClick={closeMobileSidebar}
            />
          )}
          <div
            className={`${
              mobileSidebarOpen
                ? "fixed left-0 top-14 bottom-0 z-30 block overflow-y-auto bg-white shadow-xl"
                : "hidden"
            } flex-shrink-0 md:mt-2 md:block md:static md:z-auto md:overflow-visible md:shadow-none`}
          >
            <SidebarProvider
              setPage={handleSetPage}
              defaultSelectedKey={page}
              sidebarCollapsed={isMobileViewport ? false : sidebarCollapsed}
            />
          </div>
          <main className="flex-1 min-w-0 overflow-x-hidden overflow-y-auto">{children}</main>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <LayoutContent>{children}</LayoutContent>
    </Suspense>
  );
}
