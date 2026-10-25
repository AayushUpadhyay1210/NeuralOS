import type { Metrics } from "../api/types";
import { num, pct } from "../lib/format";
import { Stat } from "./ui";

export function MetricStrip({ m }: { m: Metrics }) {
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4 lg:grid-cols-8">
      <Stat label="Avg waiting" value={num(m.avg_waiting, 1)} sub={`σ ${num(m.std_waiting, 1)}`} />
      <Stat label="Avg turnaround" value={num(m.avg_turnaround, 1)} sub={`p95 ${num(m.p95_turnaround, 1)}`} />
      <Stat label="Avg response" value={num(m.avg_response, 1)} />
      <Stat label="Throughput" value={num(m.throughput, 2)} sub="per 100 ticks" />
      <Stat label="CPU utilization" value={pct(m.cpu_utilization)} sub={`${m.idle_time} idle ticks`} />
      <Stat label="Context switches" value={m.context_switches} sub={`${m.switch_overhead} ticks overhead`} />
      <Stat label="Longest wait" value={m.max_wait}
        sub={m.starvation_count ? `${m.starvation_count} starved (> ${m.starvation_threshold})` : `none > ${m.starvation_threshold}`} />
      <Stat label="Fairness" value={m.jain_fairness.toFixed(3)} sub="Jain index, slowdown" />
    </div>
  );
}
