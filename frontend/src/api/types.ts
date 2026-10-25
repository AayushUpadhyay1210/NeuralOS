// Types mirroring the backend's JSON (backend/neuralos/api).

export type ProcessStateName = "NEW" | "READY" | "RUNNING" | "BLOCKED" | "TERMINATED";
export type SchedulerName =
  | "fcfs" | "sjf" | "srtf" | "rr" | "priority" | "priority_p" | "random" | "rl";
export type WorkloadKind =
  | "cpu_bound" | "io_bound" | "short_jobs" | "long_jobs" | "mixed_priority" | "bursty" | "mixed";
export type Distribution = "exponential" | "uniform" | "lognormal";

export interface WorkloadGenerate {
  kind: WorkloadKind;
  n_processes: number;
  seed: number;
  mean_interarrival?: number | null;
  mean_cpu_burst?: number | null;
  burst_distribution?: Distribution | null;
  priority_levels?: number;
  io_fraction?: number | null;
  mean_io_burst?: number | null;
}

export interface ProcessInput {
  pid: number;
  arrival: number;
  bursts: number[];
  priority: number;
}

export interface ProcessSpecOut extends ProcessInput {
  total_cpu: number;
  total_io: number;
}

export interface WorkloadSummary {
  n_processes: number;
  total_cpu: number;
  total_io: number;
  mean_cpu_burst: number;
  max_cpu_burst: number;
  io_processes: number;
  arrival_span: number;
  offered_load: number;
}

export interface WorkloadOut {
  config: (WorkloadGenerate & Record<string, unknown>) | null;
  processes: ProcessSpecOut[];
  summary: WorkloadSummary;
}

export type WorkloadIn = { generate: WorkloadGenerate } | { processes: ProcessInput[] };

export interface SchedulerIn {
  name: SchedulerName;
  quantum?: number;
  aging?: number | null;
  seed?: number;
  model_id?: string | null;
  checkpoint?: "best" | "final";
}

export interface Metrics {
  n_processes: number;
  makespan: number;
  avg_waiting: number;
  std_waiting: number;
  avg_turnaround: number;
  std_turnaround: number;
  avg_response: number;
  std_response: number;
  p95_turnaround: number;
  throughput: number;
  cpu_utilization: number;
  context_switches: number;
  switch_overhead: number;
  idle_time: number;
  starvation_threshold: number;
  starvation_count: number;
  max_wait: number;
  avg_slowdown: number;
  max_slowdown: number;
  jain_fairness: number;
}

export interface ProcessStats {
  pid: number;
  arrival: number;
  priority: number;
  total_cpu: number;
  total_io: number;
  cpu_bursts: number;
  start: number;
  completion: number;
  waiting: number;
  turnaround: number;
  response: number;
  max_wait: number;
  slowdown: number;
  dispatches: number;
}

export interface Segment {
  start: number;
  end: number;
  kind: "run" | "idle" | "switch";
  pid: number | null;
}

export interface TransitionOut {
  time: number;
  pid: number;
  from: ProcessStateName;
  to: ProcessStateName;
}

export interface SchedulerInfo {
  name: string;
  preemptive: boolean;
  quantum?: number;
  aging?: number | null;
  model_id?: string;
}

export interface SimulationResult {
  id: string;
  created_at: string;
  workload: WorkloadOut;
  scheduler: SchedulerInfo;
  config: { context_switch: number; max_time: number; starvation_threshold: number | null };
  metrics: Metrics;
  processes: ProcessStats[];
  timeline: Segment[];
  transitions: TransitionOut[];
  decisions: number;
}

export interface CompareResult {
  workload: WorkloadOut;
  context_switch: number;
  results: { scheduler: SchedulerInfo; metrics: Metrics; timeline: Segment[] }[];
}

export interface AppConfig {
  schedulers: { name: SchedulerName; description: string }[];
  workload_kinds: ({ name: WorkloadKind } & Record<string, unknown>)[];
  limits: Record<string, number>;
  reward_defaults: RewardWeights;
  space_defaults: { slots: number; quanta: (number | null)[] };
  seed_ranges: { train: [number, number]; validation: number; test: number };
}

export interface Health {
  status: string;
  version: string;
  rl_available: boolean;
  missing_dependencies: string[];
}

export interface RewardWeights {
  wait: number;
  fair: number;
  resp: number;
  switch: number;
}

export interface ValidationScore {
  objective: number;
  avg_waiting: number;
  avg_response: number;
  max_wait: number;
  timesteps?: number;
}

export interface ModelMeta {
  run_id: string;
  algorithm: string;
  created_at: string;
  status: "running" | "completed" | "cancelled" | "failed";
  seed: number;
  timesteps_requested: number;
  timesteps_done: number;
  reward: RewardWeights;
  space: { slots: number; quanta: (number | null)[] };
  config: Record<string, unknown>;
  versions: Record<string, string>;
  best: ValidationScore | null;
  baselines?: Record<string, ValidationScore>;
  error: string | null;
  duration_s?: number;
  has_best?: boolean;
  has_final?: boolean;
}

export interface TrainingProgress {
  run_id: string;
  status: string;
  timesteps: number;
  total_timesteps: number;
  elapsed_s: number;
  fps: number;
  curve: { timesteps: number; ep_reward_mean: number; ep_len_mean: number }[];
  validation: ValidationScore[];
  baselines: Record<string, ValidationScore>;
  best: ValidationScore | null;
}

export type TrainingState =
  | "starting" | "running" | "completed" | "cancelled" | "failed" | "interrupted" | "unknown";

export interface TrainingStatus {
  run_id: string;
  state: TrainingState;
  managed_by_api?: boolean;
  exitcode?: number | null;
  error?: string | null;
  detail?: string;
  progress?: TrainingProgress | null;
}

export interface ActiveModel {
  run_id: string;
  checkpoint: "best" | "final";
  loaded_at: string;
  algorithm: string | null;
  seed: number | null;
  timesteps: number | null;
  status: string | null;
  best: ValidationScore | null;
}

export type ModelState = "loaded" | "training" | "unavailable" | "dependencies_missing";

export interface ModelStatus {
  state: ModelState;
  active: ActiveModel | null;
  training: string[];
  missing_dependencies: string[];
  available_checkpoints: string[];
  message: string;
}

export interface Stat {
  mean: number;
  std: number;
  ci95: number;
  n: number;
}

export type SummaryMetric =
  | "avg_waiting" | "avg_turnaround" | "avg_response" | "throughput" | "cpu_utilization"
  | "context_switches" | "starvation_count" | "max_wait" | "jain_fairness" | "max_slowdown"
  | "objective";

export type SummaryGroup = { kind: string; scheduler: string } & Record<SummaryMetric, Stat>;

export interface HeadToHead {
  baseline: string;
  metric: string;
  rl_wins: number;
  ties: number;
  rl_losses: number;
  traces: number;
}

export interface BenchmarkConfig {
  name: string;
  kinds: WorkloadKind[];
  seeds: number[];
  n_processes: number;
  context_switch: number;
  rr_quantum: number;
  priority_aging: number | null;
  schedulers: string[];
  model_id: string | null;
  model_checkpoint: string;
  reward: RewardWeights;
}

export interface ExperimentMeta {
  id: string;
  config: BenchmarkConfig;
  fingerprint: string;
  created_at: string;
  duration_s: number;
}

export type ExperimentRow = { kind: string; seed: number; scheduler: string } & Record<
  string,
  number | string
>;

export interface Experiment extends ExperimentMeta {
  rows: ExperimentRow[];
  summary: { groups: SummaryGroup[]; head_to_head: HeadToHead[] };
}

export interface BenchmarkRequest {
  name: string;
  kinds: WorkloadKind[];
  n_seeds: number;
  n_processes: number;
  context_switch: number;
  rr_quantum: number;
  schedulers: SchedulerName[];
  model_id?: string | null;
  model_checkpoint?: "best" | "final";
}

export interface ReproduceResult {
  id: string;
  identical: boolean;
  rows: number;
  max_abs_difference: number;
  missing_rows: number;
}
