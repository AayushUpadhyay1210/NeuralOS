import { useMemo, useState } from "react";
import type { Experiment, SummaryGroup, SummaryMetric } from "../api/types";
import { kindLabel, METRICS, num, schedLabel } from "../lib/format";
import { MetricBars } from "./Charts";
import { Field, Provenance, Select } from "./ui";

const SUMMARY_METRICS = METRICS;

export function experimentProvenance(e: Experiment) {
  const c = e.config;
  return (
    <Provenance>
      {c.kinds.length} workload types × {c.seeds.length} seeds ({c.seeds[0]}–{c.seeds[c.seeds.length - 1]}),
      {" "}{c.n_processes} processes each, switch cost {c.context_switch}, RR quantum {c.rr_quantum}
      {c.model_id ? `, model ${c.model_id} (${c.model_checkpoint})` : ""}. Fingerprint {e.fingerprint}.
    </Provenance>
  );
}

export function ExperimentCharts({ exp }: { exp: Experiment }) {
  const kinds = useMemo(() => ["ALL", ...exp.config.kinds], [exp]);
  const [kind, setKind] = useState("ALL");
  const order = exp.config.schedulers;
  const groups = exp.summary.groups.filter((g) => g.kind === kind);
  const byName = new Map<string, SummaryGroup>(groups.map((g) => [g.scheduler, g]));
  const metrics = [
    ...SUMMARY_METRICS,
    { key: "objective", label: "RL objective (episode cost)", unit: "normalised", lowerIsBetter: true },
  ];
  return (
    <div className="flex flex-col gap-4">
      <div className="max-w-xs">
        <Field label="Workload type">
          {(id) => <Select id={id} value={kind} onChange={setKind}
            options={kinds.map((k) => ({ value: k, label: kindLabel(k) }))} />}
        </Field>
      </div>
      <div className="grid gap-x-8 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
        {metrics.map((m) => (
          <MetricBars key={m.key} title={m.label} unit={m.unit} lowerIsBetter={m.lowerIsBetter}
            format={"format" in m && m.format ? m.format : undefined}
            data={order.filter((s) => byName.has(s)).map((s) => {
              const st = byName.get(s)![m.key as SummaryMetric];
              return { key: s, label: schedLabel(s), value: st.mean, err: st.ci95, isAgent: s === "rl" };
            })} />
        ))}
      </div>
    </div>
  );
}

export function SummaryTable({ exp }: { exp: Experiment }) {
  const cols: SummaryMetric[] = ["avg_waiting", "avg_response", "avg_turnaround", "max_wait",
    "starvation_count", "jain_fairness", "context_switches", "objective"];
  const groups = exp.summary.groups.filter((g) => g.kind === "ALL");
  const best = new Map<string, number>();
  for (const c of cols) {
    const lower = c !== "jain_fairness";
    const vals = groups.map((g) => g[c].mean);
    best.set(c, lower ? Math.min(...vals) : Math.max(...vals));
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tnum">
        <thead>
          <tr className="text-left text-ink-3">
            <th className="py-2 pr-3 font-normal">Scheduler</th>
            {cols.map((c) => (
              <th key={c} className="py-2 pr-3 text-right font-normal">
                {METRICS.find((m) => m.key === c)?.label ?? "Objective"}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <tr key={g.scheduler} className="border-t border-rule">
              <td className={`py-1.5 pr-3 ${g.scheduler === "rl" ? "text-agent" : ""}`}>{schedLabel(g.scheduler)}</td>
              {cols.map((c) => (
                <td key={c} className={`py-1.5 pr-3 text-right ${best.get(c) === g[c].mean ? "font-semibold text-ink" : "text-ink-2"}`}>
                  {num(g[c].mean, c === "jain_fairness" ? 3 : 2)}
                  <span className="text-xs text-ink-3"> ±{num(g[c].std, c === "jain_fairness" ? 3 : 1)}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">Mean ± standard deviation over every trace; the best value in each column is bold.</p>
    </div>
  );
}

export function HeadToHeadTable({ exp }: { exp: Experiment }) {
  const rows = exp.summary.head_to_head;
  if (!rows.length) return null;
  const metrics = [...new Set(rows.map((r) => r.metric))];
  const baselines = [...new Set(rows.map((r) => r.baseline))];
  const label: Record<string, string> = {
    avg_waiting: "Avg waiting", avg_response: "Avg response", max_wait: "Longest wait",
    objective: "Objective",
  };
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tnum">
        <thead>
          <tr className="text-left text-ink-3">
            <th className="py-2 pr-3 font-normal">RL agent vs</th>
            {metrics.map((m) => <th key={m} className="py-2 pr-3 font-normal">{label[m] ?? m}</th>)}
          </tr>
        </thead>
        <tbody>
          {baselines.map((b) => (
            <tr key={b} className="border-t border-rule">
              <td className="py-1.5 pr-3">{schedLabel(b)}</td>
              {metrics.map((m) => {
                const r = rows.find((x) => x.baseline === b && x.metric === m)!;
                const share = r.traces ? r.rl_wins / r.traces : 0;
                return (
                  <td key={m} className="py-1.5 pr-3">
                    <span className={share > 0.5 ? "text-good" : share < 0.5 ? "text-ink-2" : ""}>
                      {r.rl_wins} won
                    </span>
                    <span className="text-ink-3">, {r.rl_losses} lost{r.ties ? `, ${r.ties} tied` : ""}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">Counts of identical traces on which the agent scored lower (better) than each baseline.</p>
    </div>
  );
}
