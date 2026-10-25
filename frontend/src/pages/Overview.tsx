import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Experiment } from "../api/types";
import { SummaryTable, experimentProvenance } from "../components/ExperimentView";
import { GanttLegend, GanttTrace } from "../components/GanttTrace";
import { MetricStrip } from "../components/MetricStrip";
import { Button, ErrorNote, ModelBadge, Page, Panel, Provenance, Stat } from "../components/ui";
import { kindLabel, schedLabel } from "../lib/format";
import { useApp } from "../state/AppState";

export function Overview() {
  const { lastSimulation: sim, setLastSimulation, modelStatus } = useApp();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [exp, setExp] = useState<Experiment | null>(null);
  const simEnd = sim?.timeline[sim.timeline.length - 1]?.end ?? 0;
  const [cursor, setCursor] = useState(simEnd);
  const [prevSim, setPrevSim] = useState(sim);
  if (sim !== prevSim) {
    setPrevSim(sim);
    setCursor(simEnd);
  }
  const [sel, setSel] = useState<[number, number] | null>(null);

  useEffect(() => {
    api.experiments().then(async (r) => {
      const pick = r.experiments.find((e) => e.config.schedulers.includes("rl")) ?? r.experiments[0];
      if (pick) setExp(await api.experiment(pick.id));
    }).catch(() => undefined);
  }, []);

  const pids = useMemo(() => (sim ? sim.processes.map((p) => p.pid).sort((a, b) => a - b) : []), [sim]);

  const quickRun = async () => {
    setBusy(true);
    setError(null);
    try {
      setLastSimulation(await api.simulate({
        workload: { generate: { kind: "mixed", n_processes: 12, seed: 7 } },
        scheduler: { name: "rr", quantum: 4 },
        context_switch: 1,
      }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Page title="NeuralOS scheduling lab"
      intro="A discrete-event CPU scheduling simulator that pits classic algorithms against a reinforcement-learning policy on identical, seeded workloads. It simulates scheduling; it is not an operating-system kernel."
      actions={modelStatus && <ModelBadge state={modelStatus.state}
        detail={modelStatus.active ? `${modelStatus.active.run_id} (${modelStatus.active.checkpoint})` : undefined} />}>
      <Panel title={sim ? `Latest run: ${schedLabel(sim.scheduler.name)}` : "Latest run"}
        aside={sim ? <GanttLegend /> : undefined}>
        {sim ? (
          <div className="flex flex-col gap-5">
            <GanttTrace timeline={sim.timeline} transitions={sim.transitions} pids={pids}
              cursor={cursor} onCursor={setCursor} selection={sel} onSelect={setSel} maxHeight={220} />
            <MetricStrip m={sim.metrics} />
            <div className="flex flex-wrap items-center justify-between gap-3">
              <Provenance>
                {sim.workload.config ? `${kindLabel(sim.workload.config.kind)}, seed ${sim.workload.config.seed}` : "Hand-built workload"},
                {" "}{sim.metrics.n_processes} processes, {sim.decisions} scheduling decisions
              </Provenance>
              <Link to="/lab" className="text-sm text-run hover:underline">Inspect in the simulation lab</Link>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-start gap-3">
            <p className="max-w-[60ch] text-ink-2">
              Nothing simulated in this session yet. Run twelve mixed processes under Round Robin to see a trace, or set up your own in the lab.
            </p>
            <div className="flex gap-2">
              <Button variant="primary" busy={busy} onClick={quickRun}>Run a sample simulation</Button>
              <Link to="/lab" className="inline-flex h-9 items-center rounded-md bg-panel-2 px-3.5 text-sm hover:bg-rule">Open the lab</Link>
            </div>
            <ErrorNote error={error} />
          </div>
        )}
      </Panel>

      <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
        <Panel title="RL agent">
          {!modelStatus ? null : (
            <div className="flex flex-col gap-4">
              <ModelBadge state={modelStatus.state} />
              <p className="text-sm text-ink-2">{modelStatus.message}</p>
              {modelStatus.active && (
                <div className="grid grid-cols-2 gap-4">
                  <Stat label="Checkpoint" value={<span className="text-agent text-lg">{modelStatus.active.run_id}</span>} sub={modelStatus.active.checkpoint} />
                  <Stat label="Training steps" value={modelStatus.active.timesteps?.toLocaleString() ?? "–"} />
                </div>
              )}
              {modelStatus.training.length > 0 && (
                <p className="text-sm">Training now: <span className="text-warn">{modelStatus.training.join(", ")}</span></p>
              )}
              <Link to="/training" className="text-sm text-run hover:underline">
                {modelStatus.active ? "Manage models" : modelStatus.available_checkpoints.length ? "Load a checkpoint" : "Train a model"}
              </Link>
            </div>
          )}
        </Panel>
        <Panel title="Latest benchmark" aside={exp ? exp.id : undefined}>
          {exp ? (
            <div className="flex flex-col gap-3">
              {experimentProvenance(exp)}
              <SummaryTable exp={exp} />
              <Link to="/experiments" className="text-sm text-run hover:underline">All experiments</Link>
            </div>
          ) : (
            <p className="text-sm text-ink-2">
              No benchmark has been saved yet. <Link to="/experiments" className="text-run hover:underline">Run one</Link> to compare schedulers across held-out seeds.
            </p>
          )}
        </Panel>
      </div>
    </Page>
  );
}
