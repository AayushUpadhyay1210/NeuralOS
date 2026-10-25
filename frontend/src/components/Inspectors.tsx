import { useMemo, useState } from "react";
import type { ProcessStateName, ProcessStats, SimulationResult } from "../api/types";
import { num } from "../lib/format";
import type { Snapshot } from "../lib/trace";
import { summariseInterval } from "../lib/trace";
import { Button } from "./ui";

const STATE_STYLE: Record<ProcessStateName, string> = {
  NEW: "text-ink-3",
  READY: "text-ink",
  RUNNING: "text-run font-medium",
  BLOCKED: "text-ink-2",
  TERMINATED: "text-ink-3",
};

function Chip({ pid, active, onClick }: { pid: number; active?: boolean; onClick?: () => void }) {
  return (
    <button type="button" onClick={onClick}
      className={`tnum rounded-md px-2 py-0.5 text-sm ${active ? "bg-run/25 text-ink" : "bg-panel-2 text-ink-2 hover:text-ink"}`}>
      P{pid}
    </button>
  );
}

/** What the CPU and queues look like at the cursor time. */
export function QueueView({ snap, pick, selected }: {
  snap: Snapshot;
  pick: (pid: number) => void;
  selected: number | null;
}) {
  const blocked = [...snap.states].filter(([, s]) => s === "BLOCKED").map(([p]) => p);
  const done = [...snap.states].filter(([, s]) => s === "TERMINATED").length;
  const waitingToArrive = [...snap.states].filter(([, s]) => s === "NEW").length;
  const cpu = snap.cpu;
  return (
    <div className="grid gap-4 sm:grid-cols-[minmax(0,12rem)_1fr]">
      <div>
        <div className="text-sm text-ink-3">CPU at t = {Math.floor(snap.time)}</div>
        <div className="mt-1 text-lg font-medium" data-testid="cpu-now">
          {cpu === null ? "Finished"
            : cpu.kind === "run" ? <span className="text-run">Running P{cpu.pid}</span>
            : cpu.kind === "switch" ? <span className="text-switch">Switching to P{cpu.pid}</span>
            : <span className="text-ink-3">Idle</span>}
        </div>
      </div>
      <div className="flex flex-col gap-2 text-sm">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-24 shrink-0 text-ink-3">Ready queue</span>
          {snap.readyQueue.length === 0 ? <span className="text-ink-3">empty</span>
            : snap.readyQueue.map((p) => <Chip key={p} pid={p} active={p === selected} onClick={() => pick(p)} />)}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="w-24 shrink-0 text-ink-3">Blocked (I/O)</span>
          {blocked.length === 0 ? <span className="text-ink-3">none</span>
            : blocked.map((p) => <Chip key={p} pid={p} active={p === selected} onClick={() => pick(p)} />)}
        </div>
        <div className="text-ink-3">
          {done} finished, {waitingToArrive} not yet arrived
        </div>
      </div>
    </div>
  );
}

type SortKey = keyof ProcessStats;

export function ProcessTable({ sim, selected, onSelect, snap }: {
  sim: SimulationResult;
  selected: number | null;
  onSelect: (pid: number) => void;
  snap: Snapshot | null;
}) {
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "pid", asc: true });
  const rows = useMemo(() => {
    const r = [...sim.processes];
    r.sort((a, b) => ((a[sort.key] as number) - (b[sort.key] as number)) * (sort.asc ? 1 : -1));
    return r;
  }, [sim, sort]);
  const thr = sim.metrics.starvation_threshold;
  const cols: { key: SortKey; label: string }[] = [
    { key: "pid", label: "Process" }, { key: "arrival", label: "Arrival" },
    { key: "total_cpu", label: "CPU burst" }, { key: "total_io", label: "I/O" },
    { key: "priority", label: "Priority" }, { key: "start", label: "First run" },
    { key: "response", label: "Response" }, { key: "waiting", label: "Waiting" },
    { key: "max_wait", label: "Longest wait" }, { key: "turnaround", label: "Turnaround" },
    { key: "completion", label: "Completion" },
  ];
  return (
    <div className="max-h-[420px] overflow-auto">
      <table className="w-full text-sm tnum">
        <thead className="sticky top-0 bg-panel">
          <tr className="text-left text-ink-3">
            {snap && <th className="px-2 py-2 font-normal">State</th>}
            {cols.map((c) => (
              <th key={c.key} className="px-2 py-2 font-normal"
                aria-sort={sort.key === c.key ? (sort.asc ? "ascending" : "descending") : "none"}>
                <button type="button" className="hover:text-ink"
                  onClick={() => setSort({ key: c.key, asc: sort.key === c.key ? !sort.asc : true })}>
                  {c.label}{sort.key === c.key ? (sort.asc ? " ↑" : " ↓") : ""}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => {
            const st = snap?.states.get(p.pid);
            return (
              <tr key={p.pid} onClick={() => onSelect(p.pid)}
                className={`cursor-pointer border-t border-rule ${p.pid === selected ? "bg-panel-2" : "hover:bg-panel-2/60"}`}>
                {snap && <td className={`px-2 py-1.5 text-xs ${st ? STATE_STYLE[st] : ""}`}>{st}</td>}
                <td className="px-2 py-1.5">
                  <button type="button" className="font-medium" onClick={() => onSelect(p.pid)}>
                    P{p.pid}
                  </button>
                </td>
                <td className="px-2 py-1.5">{p.arrival}</td>
                <td className="px-2 py-1.5">{p.total_cpu}{p.cpu_bursts > 1 ? ` (${p.cpu_bursts}×)` : ""}</td>
                <td className="px-2 py-1.5">{p.total_io || "–"}</td>
                <td className="px-2 py-1.5">{p.priority}</td>
                <td className="px-2 py-1.5">{p.start}</td>
                <td className="px-2 py-1.5">{p.response}</td>
                <td className="px-2 py-1.5">{p.waiting}</td>
                <td className={`px-2 py-1.5 ${p.max_wait > thr ? "text-warn" : ""}`}>
                  {p.max_wait}{p.max_wait > thr ? " ⚠" : ""}
                </td>
                <td className="px-2 py-1.5">{p.turnaround}</td>
                <td className="px-2 py-1.5">{p.completion}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">⚠ marks a process whose longest uninterrupted wait exceeded the starvation threshold of {thr} ticks.</p>
    </div>
  );
}

export function ProcessDetail({ sim, pid, onClose }: {
  sim: SimulationResult;
  pid: number;
  onClose: () => void;
}) {
  const p = sim.processes.find((x) => x.pid === pid);
  const spec = sim.workload.processes.find((x) => x.pid === pid);
  const history = sim.transitions.filter((t) => t.pid === pid);
  const runs = sim.timeline.filter((s) => s.kind === "run" && s.pid === pid);
  if (!p) return null;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-lg font-semibold">Process P{pid}</h3>
        <Button variant="ghost" onClick={onClose}>Close</Button>
      </div>
      <dl className="grid grid-cols-3 gap-x-4 gap-y-2 text-sm tnum">
        {[
          ["Arrival", p.arrival], ["Priority", p.priority], ["Dispatches", p.dispatches],
          ["Response", p.response], ["Waiting", p.waiting], ["Turnaround", p.turnaround],
          ["Completion", p.completion], ["Slowdown", num(p.slowdown)], ["Longest wait", p.max_wait],
        ].map(([k, v]) => (
          <div key={k as string}>
            <dt className="text-ink-3">{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {spec && (
        <p className="text-sm text-ink-2">
          Bursts:{" "}
          {spec.bursts.map((b, i) => (
            <span key={i} className={i % 2 === 0 ? "text-run" : "text-ink-3"}>
              {i % 2 === 0 ? "CPU" : "I/O"} {b}{i < spec.bursts.length - 1 ? ", " : ""}
            </span>
          ))}
        </p>
      )}
      <div>
        <h4 className="mb-1 text-sm font-semibold">Execution</h4>
        <p className="text-sm text-ink-2 tnum">
          {runs.map((r) => `${r.start}–${r.end}`).join(", ")}
        </p>
      </div>
      <div>
        <h4 className="mb-1 text-sm font-semibold">State transitions</h4>
        <ol className="max-h-56 overflow-auto text-sm tnum">
          {history.map((h, i) => (
            <li key={i} className="flex gap-3 border-t border-rule py-1">
              <span className="w-12 text-ink-3">t={h.time}</span>
              <span className={STATE_STYLE[h.from]}>{h.from}</span>
              <span className="text-ink-3" aria-label="to">›</span>
              <span className={STATE_STYLE[h.to]}>{h.to}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

export function IntervalDetail({ sim, sel, onZoom, onClear }: {
  sim: SimulationResult;
  sel: [number, number];
  onZoom: () => void;
  onClear: () => void;
}) {
  const s = summariseInterval(sim.timeline, sel[0], sel[1]);
  const len = Math.max(1, s.end - s.start);
  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-semibold tnum">Interval {s.start}–{s.end}</h3>
        <div className="flex gap-2">
          <Button onClick={onZoom}>Zoom to interval</Button>
          <Button variant="ghost" onClick={onClear}>Clear</Button>
        </div>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-sm" aria-hidden>
        <div style={{ width: `${(s.busy / len) * 100}%`, background: "var(--run)" }} />
        <div style={{ width: `${(s.switching / len) * 100}%`, background: "var(--switch)" }} />
        <div style={{ width: `${(s.idle / len) * 100}%`, background: "var(--rule)" }} />
      </div>
      <p className="tnum text-ink-2">
        {s.busy} ticks executing ({((s.busy / len) * 100).toFixed(0)}%), {s.switching} switching, {s.idle} idle.
      </p>
      <p className="tnum text-ink-2">
        Ran: {s.byPid.length === 0 ? "nothing"
          : s.byPid.map((x) => `P${x.pid} (${x.ticks})`).join(", ")}
      </p>
    </div>
  );
}
