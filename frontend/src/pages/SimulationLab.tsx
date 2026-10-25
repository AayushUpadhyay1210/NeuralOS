import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { ProcessInput, SchedulerIn, SimulationResult, WorkloadGenerate, WorkloadIn } from "../api/types";
import { GanttLegend, GanttTrace } from "../components/GanttTrace";
import { IntervalDetail, ProcessDetail, ProcessTable, QueueView } from "../components/Inspectors";
import { MetricStrip } from "../components/MetricStrip";
import { SchedulerControls, schedulerPayload } from "../components/SchedulerControls";
import { cleanWorkload, DEFAULT_WORKLOAD, WorkloadForm } from "../components/WorkloadForm";
import { Button, Empty, ErrorNote, Field, NumberInput, Page, Panel, Provenance, Select } from "../components/ui";
import { kindLabel, schedLabel } from "../lib/format";
import { snapshotAt } from "../lib/trace";
import { usePlayback } from "../lib/usePlayback";
import { useApp } from "../state/AppState";

const TEXTBOOK: ProcessInput[] = [
  { pid: 1, arrival: 0, bursts: [8], priority: 3 },
  { pid: 2, arrival: 1, bursts: [4], priority: 1 },
  { pid: 3, arrival: 2, bursts: [9], priority: 4 },
  { pid: 4, arrival: 3, bursts: [5], priority: 2 },
];

function parseBursts(s: string): number[] | null {
  const parts = s.split(/[,\s]+/).filter(Boolean).map(Number);
  if (!parts.length || parts.some((x) => !Number.isInteger(x) || x < 1)) return null;
  if (parts.length % 2 === 0) return null;
  return parts;
}

function ProcessEditor({ rows, onChange }: { rows: ProcessInput[]; onChange: (r: ProcessInput[]) => void }) {
  const [text, setText] = useState(() => rows.map((r) => r.bursts.join(", ")));
  const update = (i: number, patch: Partial<ProcessInput>) =>
    onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto">
        <table className="w-full text-sm tnum">
          <thead>
            <tr className="text-left text-ink-3">
              <th className="py-1 pr-2 font-normal">Process</th>
              <th className="py-1 pr-2 font-normal">Arrival</th>
              <th className="py-1 pr-2 font-normal">Bursts (CPU, I/O, CPU…)</th>
              <th className="py-1 pr-2 font-normal">Priority</th>
              <th className="sr-only">Remove</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const bad = parseBursts(text[i] ?? "") === null;
              return (
                <tr key={i}>
                  <td className="py-1 pr-2">P{r.pid}</td>
                  <td className="py-1 pr-2 w-24">
                    <NumberInput value={r.arrival} min={0} onChange={(v) => update(i, { arrival: v })} />
                  </td>
                  <td className="py-1 pr-2">
                    <input aria-label={`Bursts for P${r.pid}`} aria-invalid={bad}
                      className={`h-9 w-full rounded-md border bg-ground px-2.5 ${bad ? "border-bad" : "border-rule"}`}
                      value={text[i] ?? ""}
                      onChange={(e) => {
                        const nt = [...text];
                        nt[i] = e.target.value;
                        setText(nt);
                        const b = parseBursts(e.target.value);
                        if (b) update(i, { bursts: b });
                      }} />
                  </td>
                  <td className="py-1 pr-2 w-20">
                    <NumberInput value={r.priority} min={0} onChange={(v) => update(i, { priority: v })} />
                  </td>
                  <td className="py-1">
                    <Button variant="ghost" aria-label={`Remove P${r.pid}`}
                      disabled={rows.length <= 1}
                      onClick={() => {
                        setText(text.filter((_, j) => j !== i));
                        onChange(rows.filter((_, j) => j !== i));
                      }}>Remove</Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => {
          const pid = Math.max(0, ...rows.map((r) => r.pid)) + 1;
          setText([...text, "3"]);
          onChange([...rows, { pid, arrival: 0, bursts: [3], priority: 0 }]);
        }}>Add process</Button>
        <Button variant="ghost" onClick={() => {
          setText(TEXTBOOK.map((r) => r.bursts.join(", ")));
          onChange(TEXTBOOK);
        }}>Load textbook example</Button>
      </div>
      <p className="text-xs text-ink-3">An odd number of bursts, alternating CPU and I/O, starting and ending with CPU. Lower priority number runs first.</p>
    </div>
  );
}

export function SimulationLab() {
  const { lastSimulation, setLastSimulation, pendingWorkload, refreshModelStatus } = useApp();
  const [mode, setMode] = useState<"generate" | "custom">("generate");
  const [wl, setWl] = useState<WorkloadGenerate>(pendingWorkload ?? DEFAULT_WORKLOAD);
  const [custom, setCustom] = useState<ProcessInput[]>(TEXTBOOK);
  const [sched, setSched] = useState<SchedulerIn>({ name: "rr", quantum: 4 });
  const [cs, setCs] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [sim, setSim] = useState<SimulationResult | null>(lastSimulation);
  const [selectedPid, setSelectedPid] = useState<number | null>(null);
  const [selection, setSelection] = useState<[number, number] | null>(null);
  const [view, setView] = useState<[number, number] | undefined>(undefined);
  const pb = usePlayback(sim);


  const pids = useMemo(() => (sim ? sim.processes.map((p) => p.pid).sort((a, b) => a - b) : []), [sim]);
  const snap = useMemo(
    () => (sim ? snapshotAt(pb.t, sim.transitions, sim.timeline, pids) : null),
    [sim, pb.t, pids],
  );

  const runSim = async () => {
    setBusy(true);
    setError(null);
    try {
      const workload: WorkloadIn = mode === "generate"
        ? { generate: cleanWorkload(wl) } : { processes: custom };
      const res = await api.simulate({ workload, scheduler: schedulerPayload(sched), context_switch: cs });
      setSim(res);
      setLastSimulation(res);
      setSelectedPid(null);
      setSelection(null);
      setView(undefined);
    } catch (e) {
      setError(e);
      void refreshModelStatus();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Page title="Simulation lab"
      intro="Configure a workload, pick a scheduler, and step through the exact trace the simulator produced.">
      <Panel title="Setup">
        <div className="flex flex-col gap-5">
          <div className="flex gap-1 rounded-md bg-ground p-1 self-start" role="tablist" aria-label="Workload source">
            {(["generate", "custom"] as const).map((m) => (
              <button key={m} type="button" role="tab" aria-selected={mode === m}
                onClick={() => setMode(m)}
                className={`rounded px-3 py-1.5 text-sm ${mode === m ? "bg-panel-2 text-ink" : "text-ink-3 hover:text-ink"}`}>
                {m === "generate" ? "Generated workload" : "Hand-built processes"}
              </button>
            ))}
          </div>
          {mode === "generate" ? <WorkloadForm value={wl} onChange={setWl} />
            : <ProcessEditor rows={custom} onChange={setCustom} />}
          <SchedulerControls value={sched} onChange={setSched} contextSwitch={cs} onContextSwitch={setCs} />
          <div className="flex items-center gap-3">
            <Button variant={sched.name === "rl" ? "agent" : "primary"} busy={busy} onClick={runSim}>
              Run simulation
            </Button>
          </div>
          <ErrorNote error={error} />
        </div>
      </Panel>

      {!sim ? (
        <Empty title="No simulation yet">
          Run a simulation above. The trace, queue view and process inspector fill in from the result.
        </Empty>
      ) : (
        <>
          <Panel title={`${schedLabel(sim.scheduler.name)}${sim.scheduler.quantum ? `, quantum ${sim.scheduler.quantum}` : ""}`}
            aside={<Provenance>
              {sim.workload.config ? `${kindLabel(sim.workload.config.kind)}, seed ${sim.workload.config.seed}` : "Hand-built workload"},{" "}
              {sim.metrics.n_processes} processes, switch cost {sim.config.context_switch}
              {sim.scheduler.model_id ? `, model ${sim.scheduler.model_id}` : ""}
            </Provenance>}>
            <MetricStrip m={sim.metrics} />
          </Panel>

          <Panel title="Gantt trace" aside={<GanttLegend />}>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {pb.playing ? <Button onClick={pb.pause}>Pause</Button>
                : <Button variant="primary" onClick={pb.play}>{pb.t >= pb.end ? "Play from start" : "Play"}</Button>}
              <Button onClick={() => pb.step(-1)} aria-label="Step back to previous event">Step back</Button>
              <Button onClick={() => pb.step(1)} aria-label="Step to next event">Step</Button>
              <Button variant="ghost" onClick={pb.reset}>Reset</Button>
              <Button variant="ghost" onClick={pb.replay}>Replay</Button>
              <div className="ml-auto flex items-center gap-2 text-sm">
                <Field label="Speed">
                  {(id) => (
                    <Select id={id} value={String(pb.speed)} onChange={(v) => pb.setSpeed(Number(v))}
                      options={[2, 5, 10, 25, 60].map((s) => ({ value: String(s), label: `${s} ticks/s` }))} />
                  )}
                </Field>
                <Field label="Time">
                  {(id) => <NumberInput id={id} value={pb.tInt} min={0} max={pb.end} onChange={(v) => pb.seek(v)} />}
                </Field>
              </div>
            </div>
            <input type="range" aria-label="Playback position" className="mb-3 w-full accent-[var(--run)]"
              min={0} max={pb.end} value={pb.tInt} onChange={(e) => pb.seek(Number(e.target.value))} />
            <GanttTrace timeline={sim.timeline} transitions={sim.transitions} pids={pids}
              cursor={pb.t} onCursor={pb.seek} selection={selection} onSelect={setSelection}
              view={view} highlightPid={selectedPid} onPickPid={setSelectedPid} onStep={pb.step} />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-3">
              <span>Click to move the cursor, drag to select an interval, use arrow keys to step.</span>
              {view && <Button variant="ghost" onClick={() => setView(undefined)}>Show full timeline</Button>}
            </div>
            {selection && (
              <div className="mt-4 border-t border-rule pt-4">
                <IntervalDetail sim={sim} sel={selection} onZoom={() => setView(selection)}
                  onClear={() => setSelection(null)} />
              </div>
            )}
          </Panel>

          {snap && (
            <Panel title="Queues at the cursor">
              <QueueView snap={snap} pick={setSelectedPid} selected={selectedPid} />
            </Panel>
          )}

          <div className={`grid gap-6 ${selectedPid !== null ? "lg:grid-cols-[1fr_360px]" : ""}`}>
            <Panel title="Process inspector" aside="Select a row for its history">
              <ProcessTable sim={sim} selected={selectedPid} onSelect={setSelectedPid} snap={snap} />
            </Panel>
            {selectedPid !== null && (
              <Panel>
                <ProcessDetail sim={sim} pid={selectedPid} onClose={() => setSelectedPid(null)} />
              </Panel>
            )}
          </div>
        </>
      )}
    </Page>
  );
}
