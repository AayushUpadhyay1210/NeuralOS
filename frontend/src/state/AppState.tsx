import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api/client";
import type { AppConfig, Health, ModelStatus, SimulationResult, WorkloadGenerate } from "../api/types";

interface AppState {
  health: Health | null;
  config: AppConfig | null;
  apiError: string | null;
  modelStatus: ModelStatus | null;
  refreshModelStatus: () => Promise<void>;
  lastSimulation: SimulationResult | null;
  setLastSimulation: (s: SimulationResult | null) => void;
  /** Workload handed from the workload explorer to the simulation lab. */
  pendingWorkload: WorkloadGenerate | null;
  setPendingWorkload: (w: WorkloadGenerate | null) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [lastSimulation, setLastSimulation] = useState<SimulationResult | null>(null);
  const [pendingWorkload, setPendingWorkload] = useState<WorkloadGenerate | null>(null);

  const refreshModelStatus = useCallback(async () => {
    try {
      setModelStatus(await api.modelStatus());
    } catch {
      /* surfaced through apiError by the health poll */
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const boot = async () => {
      try {
        const [h, c] = await Promise.all([api.health(), api.config()]);
        if (!alive) return;
        setHealth(h);
        setConfig(c);
        setApiError(null);
        await refreshModelStatus();
      } catch (e) {
        if (alive) setApiError(e instanceof Error ? e.message : String(e));
      }
    };
    void boot();
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") void boot();
    }, 8000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [refreshModelStatus]);

  const value = useMemo(
    () => ({ health, config, apiError, modelStatus, refreshModelStatus, lastSimulation,
      setLastSimulation, pendingWorkload, setPendingWorkload }),
    [health, config, apiError, modelStatus, refreshModelStatus, lastSimulation, pendingWorkload],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp must be used inside AppStateProvider");
  return v;
}
