import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CompareResult, Experiment, ExperimentMeta, SchedulerName, WorkloadGenerate } from "../api/types";
import { MetricBars } from "../components/Charts";
import { ExperimentCharts, experimentProvenance, HeadToHeadTable } from "../components/ExperimentView";
import { cleanWorkload, WorkloadForm } from "../components/WorkloadForm";
import { Button, Checkbox, Empty, ErrorNote, Field, Loading, NumberInput, Page, Panel, Provenance, Select } from "../components/ui";
import { kindLabel, METRICS, schedLabel } from "../lib/format";
import { useApp } from "../state/AppState";

const ALL: SchedulerName[] = ["fcfs", "sjf", "srtf", "rr", "priority", "priority_p", "random", "rl"];

function SingleTrace() {
  const { modelStatus } = useApp();
  const rlReady = modelStatus?.state === "loaded";
  const [wl, setWl] = useState<WorkloadGenerate>({ kind: "mixed", n_processes: 40, seed: 10_000_001 });
  const [chosen, setChosen] = useState<SchedulerName[]>(["fcfs", "sjf", "srtf", "rr", "priority"]);
  const [quantum, setQuantum] = useState(4);
  const [cs, setCs] = useState(1);
  const [res, setRes] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setRes(await api.compare({
        workload: { generate: cleanWorkload(wl) },
        schedulers: chosen.filter((n) => n !== "rl" || rlReady).map((name) => ({ name, quantum })),
        context_switch: cs,
      }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="One workload, every scheduler"
      aside="Each scheduler runs on the identical trace">
      <div className="flex flex-col gap-4">
        <WorkloadForm value={wl} onChange={setWl} />
        <div className="flex flex-wrap gap-x-5 gap-y-2" role="group" aria-label="Schedulers to compare">
          {ALL.map((n) => (
            <Checkbox key={n} checked={chosen.includes(n) && (n !== "rl" || rlReady)}
              disabled={n === "rl" && !rlReady}
              label={n === "rl" && !rlReady ? "RL agent (no model loaded)" : schedLabel(n)}
              onChange={(v) => setChosen(v ? [...chosen, n] : chosen.filter((x) => x !== n))} />
          ))}
        </div>
        <div className="grid max-w-md grid-cols-2 gap-3">
          <Field label="Round Robin quantum">
            {(id) => <NumberInput id={id} value={quantum} min={1} onChange={setQuantum} />}
          </Field>
          <Field label="Context switch cost">
            {(id) => <NumberInput id={id} value={cs} min={0} max={20} onChange={setCs} />}
          </Field>
        </div>
        <div>
          <Button variant="primary" busy={busy} disabled={chosen.length === 0} onClick={run}>
            Compare schedulers
          </Button>
        </div>
        <ErrorNote error={error} />
        {res && (
          <>
            <Provenance>
              {kindLabel(res.workload.config?.kind ?? "")}, seed {res.workload.config?.seed},{" "}
              {res.workload.summary.n_processes} processes, offered load {res.workload.summary.offered_load.toFixed(2)},
              switch cost {res.context_switch}. A single trace; see the multi-seed results below before drawing conclusions.
            </Provenance>
            <div className="grid gap-x-8 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
              {METRICS.map((m) => (
                <MetricBars key={m.key} title={m.label} unit={m.unit} lowerIsBetter={m.lowerIsBetter}
                  format={m.format}
                  data={res.results.map((r) => ({
                    key: r.scheduler.name,
                    label: schedLabel(r.scheduler.name),
                    value: r.metrics[m.key as keyof typeof r.metrics] as number,
                    isAgent: r.scheduler.name === "rl",
                  }))} />
              ))}
            </div>
          </>
        )}
      </div>
    </Panel>
  );
}

function MultiSeed() {
  const [list, setList] = useState<ExperimentMeta[] | null>(null);
  const [sel, setSel] = useState<string>("");
  const [exp, setExp] = useState<Experiment | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    api.experiments().then((r) => {
      setList(r.experiments);
      const withRl = r.experiments.find((e) => e.config.schedulers.includes("rl"));
      setSel((withRl ?? r.experiments[0])?.id ?? "");
    }).catch(setError);
  }, []);
  useEffect(() => {
    if (sel) api.experiment(sel).then(setExp).catch(setError);
  }, [sel]);
  const shown = exp && exp.id === sel ? exp : null;
  return (
    <Panel title="Across many seeds" aside="Saved benchmark experiments">
      <ErrorNote error={error} />
      {list === null && !error ? <Loading label="Loading experiments" />
        : list && list.length === 0 ? (
          <Empty title="No benchmark experiments saved">
            Run one on the Experiments page. Benchmarks use held-out test seeds (10,000,000 and up) that training never saw.
          </Empty>
        ) : list && (
          <div className="flex flex-col gap-5">
            <div className="max-w-xl">
              <Field label="Experiment">
                {(id) => <Select id={id} value={sel} onChange={setSel}
                  options={list.map((e) => ({ value: e.id, label: e.id }))} />}
              </Field>
            </div>
            {!shown ? <Loading label="Loading results" /> : (
              <>
                {experimentProvenance(shown)}
                <ExperimentCharts exp={shown} />
                {shown.summary.head_to_head.length > 0 && (
                  <div>
                    <h3 className="mb-2 font-semibold">Head to head</h3>
                    <HeadToHeadTable exp={shown} />
                  </div>
                )}
              </>
            )}
          </div>
        )}
    </Panel>
  );
}

export function Comparison() {
  return (
    <Page title="Algorithm comparison"
      intro="Every number here comes from the simulator. Single-trace comparisons are quick to explore; the multi-seed experiments are the ones to trust.">
      <SingleTrace />
      <MultiSeed />
    </Page>
  );
}
