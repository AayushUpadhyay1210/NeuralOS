import type {
  AppConfig,
  BenchmarkRequest,
  CompareResult,
  Experiment,
  ExperimentMeta,
  Health,
  ModelMeta,
  ModelStatus,
  ReproduceResult,
  RewardWeights,
  SchedulerIn,
  SimulationResult,
  TrainingProgress,
  TrainingStatus,
  WorkloadGenerate,
  WorkloadIn,
  WorkloadOut,
} from "./types";

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Turn FastAPI error bodies (string or validation list) into one readable line. */
function describe(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => {
        const where = (d.loc ?? []).filter((x) => x !== "body").join(".");
        return where ? `${where}: ${d.msg}` : String(d.msg);
      })
      .join("; ");
  }
  return "Unexpected response from the API";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      0,
      "Cannot reach the NeuralOS API. Start it with: uvicorn neuralos.api.app:app --port 8000",
    );
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body.detail !== undefined) msg = describe(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/api/health"),
  config: () => request<AppConfig>("/api/config"),
  generateWorkload: (cfg: WorkloadGenerate) => post<WorkloadOut>("/api/workloads/generate", cfg),
  simulate: (body: { workload: WorkloadIn; scheduler: SchedulerIn; context_switch: number }) =>
    post<SimulationResult>("/api/simulations", body),
  getSimulation: (id: string) => request<SimulationResult>(`/api/simulations/${id}`),
  compare: (body: { workload: WorkloadIn; schedulers: SchedulerIn[]; context_switch: number }) =>
    post<CompareResult>("/api/compare", body),
  models: () => request<{ models: ModelMeta[] }>("/api/models"),
  modelStatus: () => request<ModelStatus>("/api/models/status"),
  model: (id: string) =>
    request<{ metadata: ModelMeta; progress: TrainingProgress | null; training: TrainingStatus }>(
      `/api/models/${encodeURIComponent(id)}`,
    ),
  loadModel: (id: string, checkpoint: "best" | "final") =>
    post<unknown>(`/api/models/${encodeURIComponent(id)}/load`, { checkpoint }),
  unloadModel: () => post<unknown>("/api/models/unload"),
  evaluateModel: (id: string, body: { checkpoint: "best" | "final"; n_seeds: number;
    n_processes: number }) => post<Experiment>(`/api/models/${encodeURIComponent(id)}/evaluate`, body),
  startTraining: (body: { run_id?: string; timesteps: number; seed: number;
    reward?: Partial<RewardWeights>; n_envs?: number }) =>
    post<{ run_id: string; state: string }>("/api/training", body),
  trainingStatus: (id: string) =>
    request<TrainingStatus>(`/api/training/${encodeURIComponent(id)}`),
  cancelTraining: (id: string) => post<unknown>(`/api/training/${encodeURIComponent(id)}/cancel`),
  experiments: () => request<{ experiments: ExperimentMeta[] }>("/api/experiments"),
  experiment: (id: string) => request<Experiment>(`/api/experiments/${encodeURIComponent(id)}`),
  runExperiment: (body: BenchmarkRequest) => post<Experiment>("/api/experiments", body),
  reproduce: (id: string) =>
    post<ReproduceResult>(`/api/experiments/${encodeURIComponent(id)}/reproduce`),
  exportUrl: (id: string, format: "csv" | "json") =>
    `${API_BASE}/api/experiments/${encodeURIComponent(id)}/export?format=${format}`,
};
