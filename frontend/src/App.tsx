import { lazy, Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { Loading, ModelBadge } from "./components/ui";

// Chart-heavy pages load on demand so the first paint stays small.
const Overview = lazy(() => import("./pages/Overview").then((m) => ({ default: m.Overview })));
const SimulationLab = lazy(() => import("./pages/SimulationLab").then((m) => ({ default: m.SimulationLab })));
const Workloads = lazy(() => import("./pages/Workloads").then((m) => ({ default: m.Workloads })));
const Comparison = lazy(() => import("./pages/Comparison").then((m) => ({ default: m.Comparison })));
const Training = lazy(() => import("./pages/Training").then((m) => ({ default: m.Training })));
const Experiments = lazy(() => import("./pages/Experiments").then((m) => ({ default: m.Experiments })));
const Docs = lazy(() => import("./pages/Docs").then((m) => ({ default: m.Docs })));
import { useApp } from "./state/AppState";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/lab", label: "Simulation lab" },
  { to: "/workloads", label: "Workloads" },
  { to: "/compare", label: "Comparison" },
  { to: "/training", label: "AI training" },
  { to: "/experiments", label: "Experiments" },
  { to: "/docs", label: "Documentation" },
];

function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      return (localStorage.getItem("neuralos-theme") as "dark" | "light") ?? "dark";
    } catch {
      return "dark";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("neuralos-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return (
    <button type="button" className="text-sm text-ink-3 hover:text-ink"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
      {theme === "dark" ? "Light theme" : "Dark theme"}
    </button>
  );
}

function Mark() {
  // Three execution bars: the product's own subject, drawn as a tiny trace.
  return (
    <svg width="22" height="18" viewBox="0 0 22 18" aria-hidden>
      <rect x="0" y="1" width="9" height="4" rx="1" fill="var(--run)" />
      <rect x="9" y="7" width="12" height="4" rx="1" fill="var(--agent)" />
      <rect x="4" y="13" width="7" height="4" rx="1" fill="var(--run)" />
    </svg>
  );
}

export default function App() {
  const { apiError, modelStatus, health } = useApp();
  const [menu, setMenu] = useState(false);
  return (
    <div className="min-h-screen md:grid md:grid-cols-[220px_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:bg-panel focus:p-2">Skip to content</a>
      <aside className="border-rule md:sticky md:top-0 md:h-screen md:self-start md:overflow-y-auto md:border-r">
        <div className="flex items-center justify-between px-5 py-4">
          <NavLink to="/" className="flex items-center gap-2.5">
            <Mark />
            <span className="text-lg font-semibold" style={{ fontStretch: "125%" }}>NeuralOS</span>
          </NavLink>
          <button type="button" className="text-sm text-ink-2 md:hidden" aria-expanded={menu}
            aria-controls="nav" onClick={() => setMenu(!menu)}>Menu</button>
        </div>
        <nav id="nav" aria-label="Main" className={`${menu ? "block" : "hidden"} px-3 pb-4 md:block`}>
          <ul className="flex flex-col gap-0.5">
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink to={n.to} end={n.to === "/"} onClick={() => setMenu(false)}
                  className={({ isActive }) => `block rounded-md px-3 py-2 text-[0.95rem] ${isActive ? "bg-panel text-ink" : "text-ink-2 hover:text-ink"}`}>
                  {n.label}
                </NavLink>
              </li>
            ))}
          </ul>
          <div className="mt-6 flex flex-col gap-3 border-t border-rule px-3 pt-4">
            {modelStatus && <ModelBadge state={modelStatus.state} />}
            <span className="text-xs text-ink-3">
              {apiError ? "API offline" : health ? `API v${health.version} connected` : "Connecting to API…"}
            </span>
            <ThemeToggle />
          </div>
        </nav>
      </aside>
      <main id="main" className="min-w-0">
        {apiError && (
          <div role="alert" className="border-b border-bad/40 bg-bad/10 px-4 py-3 text-sm md:px-8">
            <strong className="text-bad">The API is not reachable.</strong> {apiError}
          </div>
        )}
        <Suspense fallback={<div className="px-8"><Loading /></div>}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/lab" element={<SimulationLab />} />
          <Route path="/workloads" element={<Workloads />} />
          <Route path="/compare" element={<Comparison />} />
          <Route path="/training" element={<Training />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="*" element={<div className="p-8">Page not found. <NavLink className="text-run" to="/">Go to the overview</NavLink>.</div>} />
        </Routes>
        </Suspense>
      </main>
    </div>
  );
}
