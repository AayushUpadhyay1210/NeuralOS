export const SCHEDULER_LABELS: Record<string, string> = {
  fcfs: "FCFS",
  sjf: "SJF",
  srtf: "SRTF",
  rr: "Round Robin",
  priority: "Priority",
  priority_p: "Priority (preemptive)",
  random: "Random",
  rl: "RL agent",
};

export const KIND_LABELS: Record<string, string> = {
  cpu_bound: "CPU-bound",
  io_bound: "I/O-bound",
  short_jobs: "Short jobs",
  long_jobs: "Long jobs",
  mixed_priority: "Mixed priority",
  bursty: "Bursty arrivals",
  mixed: "Mixed",
  ALL: "All workloads",
};

export const schedLabel = (n: string) => SCHEDULER_LABELS[n] ?? n;
export const kindLabel = (k: string) => KIND_LABELS[k] ?? k;

export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  if (Number.isInteger(x)) return x.toLocaleString();
  return x.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export function duration(s: number): string {
  if (s < 60) return `${s.toFixed(0)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
}

export interface MetricDef {
  key: string;
  label: string;
  unit: string;
  lowerIsBetter: boolean;
  format?: (x: number) => string;
}

export const METRICS: MetricDef[] = [
  { key: "avg_waiting", label: "Average waiting time", unit: "ticks", lowerIsBetter: true },
  { key: "avg_turnaround", label: "Average turnaround", unit: "ticks", lowerIsBetter: true },
  { key: "avg_response", label: "Average response time", unit: "ticks", lowerIsBetter: true },
  { key: "throughput", label: "Throughput", unit: "per 100 ticks", lowerIsBetter: false },
  { key: "cpu_utilization", label: "CPU utilization", unit: "", lowerIsBetter: false, format: pct },
  { key: "jain_fairness", label: "Fairness (Jain index of slowdown)", unit: "0–1",
    lowerIsBetter: false, format: (x) => x.toFixed(3) },
  { key: "max_wait", label: "Longest single wait", unit: "ticks", lowerIsBetter: true },
  { key: "starvation_count", label: "Starved processes", unit: "count", lowerIsBetter: true },
  { key: "context_switches", label: "Context switches", unit: "count", lowerIsBetter: true },
];
