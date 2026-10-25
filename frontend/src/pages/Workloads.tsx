import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api } from "../api/client";
import type { WorkloadGenerate, WorkloadKind, WorkloadOut } from "../api/types";
import { MetricBars } from "../components/Charts";
import { cleanWorkload, KIND_OPTIONS, WorkloadForm } from "../components/WorkloadForm";
import { Button, ErrorNote, Page, Panel, Provenance, Stat } from "../components/ui";
import { kindLabel, num } from "../lib/format";
import { useApp } from "../state/AppState";

const AXIS = { stroke: "var(--rule)", tick: { fill: "var(--ink-3)", fontSize: 11 }, tickLine: false };

function histogram(values: number[], bins = 12) {
  if (!values.length) return [];
  const max = Math.max(...values);
  const width = Math.max(1, Math.ceil(max / bins));
  const counts = new Array<number>(Math.ceil((max + 1) / width)).fill(0);
  for (const v of values) counts[Math.floor(v / width)]! += 1;
  return counts.map((c, i) => ({ range: `${i * width}–${(i + 1) * width - 1}`, count: c }));
}

function WorkloadCharts({ w }: { w: WorkloadOut }) {
  const cpu = w.processes.flatMap((p) => p.bursts.filter((_, i) => i % 2 === 0));
  const points = w.processes.map((p) => ({ arrival: p.arrival, cpu: p.total_cpu, pid: p.pid, io: p.total_io }));
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <figure>
        <figcaption className="text-sm font-medium">Arrivals and CPU demand</figcaption>
        <div className="text-xs text-ink-3">Each dot is a process: arrival time against total CPU work</div>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} />
              <XAxis dataKey="arrival" type="number" name="Arrival" {...AXIS} />
              <YAxis dataKey="cpu" type="number" name="CPU" {...AXIS} width={44} />
              <ZAxis range={[48, 48]} />
              <Tooltip content={({ active, payload }) => {
                const d = active && payload?.[0] ? (payload[0].payload as typeof points[number]) : null;
                return d ? (
                  <div className="rounded-md bg-panel-2 px-2.5 py-1.5 text-xs tnum">
                    <div className="font-medium">P{d.pid}</div>
                    <div className="text-ink-2">arrives {d.arrival}, CPU {d.cpu}, I/O {d.io}</div>
                  </div>
                ) : null;
              }} />
              <Scatter data={points} fill="var(--run)" stroke="var(--panel)" strokeWidth={2} isAnimationActive={false} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </figure>
      <figure>
        <figcaption className="text-sm font-medium">CPU burst lengths</figcaption>
        <div className="text-xs text-ink-3">Histogram of every CPU burst, in ticks</div>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogram(cpu)} margin={{ top: 10, right: 10, left: -10, bottom: 0 }} barCategoryGap={2}>
              <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={0.6} />
              <XAxis dataKey="range" {...AXIS} fontSize={10} />
              <YAxis {...AXIS} allowDecimals={false} width={36} />
              <Tooltip cursor={{ fill: "var(--panel-2)" }} content={({ active, payload }) => {
                const d = active && payload?.[0] ? (payload[0].payload as { range: string; count: number }) : null;
                return d ? <div className="rounded-md bg-panel-2 px-2.5 py-1.5 text-xs tnum">{d.count} bursts of {d.range} ticks</div> : null;
              }} />
              <Bar dataKey="count" fill="var(--run)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </figure>
    </div>
  );
}

export function Workloads() {
  const nav = useNavigate();
  const { setPendingWorkload } = useApp();
  const [wl, setWl] = useState<WorkloadGenerate>({ kind: "bursty", n_processes: 40, seed: 3 });
  const [res, setRes] = useState<WorkloadOut | null>(null);
  const [kinds, setKinds] = useState<{ kind: WorkloadKind; w: WorkloadOut }[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const gen = async () => {
    setBusy("gen");
    setError(null);
    try {
      setRes(await api.generateWorkload(cleanWorkload(wl)));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };
  const compareKinds = async () => {
    setBusy("kinds");
    setError(null);
    try {
      const all = await Promise.all(KIND_OPTIONS.map(async (k) => ({
        kind: k.value,
        w: await api.generateWorkload({ kind: k.value, n_processes: wl.n_processes, seed: wl.seed }),
      })));
      setKinds(all);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Page title="Workload explorer"
      intro="Generate seeded process traces and see how the workload types differ before you schedule them.">
      <Panel title="Generator">
        <div className="flex flex-col gap-4">
          <WorkloadForm value={wl} onChange={setWl} />
          <div className="flex flex-wrap gap-2">
            <Button variant="primary" busy={busy === "gen"} onClick={gen}>Generate workload</Button>
            <Button busy={busy === "kinds"} onClick={compareKinds}>Compare all types at this size and seed</Button>
            {res && (
              <Button variant="ghost" onClick={() => { setPendingWorkload(cleanWorkload(wl)); nav("/lab"); }}>
                Open in simulation lab
              </Button>
            )}
          </div>
          <ErrorNote error={error} />
        </div>
      </Panel>

      {res && (
        <Panel title={`${kindLabel(res.config?.kind ?? "")}, seed ${res.config?.seed}`}
          aside={<Provenance>Generated by the backend from this seed; regenerating returns the identical trace.</Provenance>}>
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-2 gap-5 sm:grid-cols-5">
              <Stat label="Processes" value={res.summary.n_processes} />
              <Stat label="Mean CPU burst" value={num(res.summary.mean_cpu_burst, 1)} sub={`max ${res.summary.max_cpu_burst}`} />
              <Stat label="Doing I/O" value={res.summary.io_processes} sub={`${res.summary.total_io} I/O ticks`} />
              <Stat label="Arrival span" value={res.summary.arrival_span} sub="ticks" />
              <Stat label="Offered load" value={num(res.summary.offered_load, 2)} sub="CPU work ÷ arrival span" />
            </div>
            <WorkloadCharts w={res} />
            <details>
              <summary className="cursor-pointer text-sm text-ink-2">Show all {res.processes.length} processes</summary>
              <div className="mt-2 max-h-80 overflow-auto">
                <table className="w-full text-sm tnum">
                  <thead><tr className="text-left text-ink-3">
                    <th className="py-1 font-normal">Process</th><th className="font-normal">Arrival</th>
                    <th className="font-normal">Priority</th><th className="font-normal">Bursts</th>
                  </tr></thead>
                  <tbody>
                    {res.processes.map((p) => (
                      <tr key={p.pid} className="border-t border-rule">
                        <td className="py-1">P{p.pid}</td><td>{p.arrival}</td><td>{p.priority}</td>
                        <td className="text-ink-2">{p.bursts.join(", ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        </Panel>
      )}

      {kinds && (
        <Panel title="Workload types side by side" aside={`${wl.n_processes} processes, seed ${wl.seed}`}>
          <div className="grid gap-x-8 gap-y-8 md:grid-cols-2">
            {([
              ["offered_load", "Offered load", "CPU work ÷ arrival span"],
              ["mean_cpu_burst", "Mean CPU burst", "ticks"],
              ["io_processes", "Processes doing I/O", "count"],
              ["arrival_span", "Arrival span", "ticks"],
            ] as const).map(([key, title, unit]) => (
              <MetricBars key={key} title={title} unit={unit} lowerIsBetter={false} neutral
                data={kinds.map((k) => ({ key: k.kind, label: kindLabel(k.kind), value: k.w.summary[key] }))} />
            ))}
          </div>
        </Panel>
      )}
    </Page>
  );
}
