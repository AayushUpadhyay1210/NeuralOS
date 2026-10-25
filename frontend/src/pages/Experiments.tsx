import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { BenchmarkRequest, Experiment, ExperimentMeta, ReproduceResult, SchedulerName, WorkloadKind } from "../api/types";
import { ExperimentCharts, experimentProvenance, HeadToHeadTable, SummaryTable } from "../components/ExperimentView";
import { KIND_OPTIONS } from "../components/WorkloadForm";
import { Button, Checkbox, Empty, ErrorNote, Field, Loading, NumberInput, Page, Panel, Select, TextInput } from "../components/ui";
import { num, schedLabel } from "../lib/format";
import { useApp } from "../state/AppState";

const SCHEDS: SchedulerName[] = ["fcfs", "sjf", "srtf", "rr", "priority", "priority_p", "random", "rl"];

function RunForm({ onDone }: { onDone: (e: Experiment) => void }) {
  const { modelStatus } = useApp();
  const ckpts = modelStatus?.available_checkpoints ?? [];
  const [req, setReq] = useState<BenchmarkRequest>({
    name: "benchmark", kinds: KIND_OPTIONS.map((k) => k.value), n_seeds: 5, n_processes: 30,
    context_switch: 1, rr_quantum: 4, schedulers: ["fcfs", "sjf", "srtf", "rr", "priority", "random"],
    model_id: null,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const useRl = req.schedulers.includes("rl");
  const runs = req.kinds.length * req.n_seeds * req.schedulers.length;
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      onDone(await api.runExperiment({ ...req, model_id: useRl ? (req.model_id || ckpts[0]) : null }));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Field label="Name">{(id) => <TextInput id={id} value={req.name} onChange={(v) => setReq({ ...req, name: v })} />}</Field>
        <Field label="Seeds per type" hint="Test seeds 10,000,000+">
          {(id) => <NumberInput id={id} value={req.n_seeds} min={1} max={50} onChange={(v) => setReq({ ...req, n_seeds: v })} />}
        </Field>
        <Field label="Processes">{(id) => <NumberInput id={id} value={req.n_processes} min={1} max={300} onChange={(v) => setReq({ ...req, n_processes: v })} />}</Field>
        <Field label="Switch cost">{(id) => <NumberInput id={id} value={req.context_switch} min={0} max={20} onChange={(v) => setReq({ ...req, context_switch: v })} />}</Field>
        <Field label="RR quantum">{(id) => <NumberInput id={id} value={req.rr_quantum} min={1} onChange={(v) => setReq({ ...req, rr_quantum: v })} />}</Field>
      </div>
      <fieldset>
        <legend className="mb-1.5 text-sm text-ink-2">Workload types</legend>
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          {KIND_OPTIONS.map((k) => (
            <Checkbox key={k.value} label={k.label} checked={req.kinds.includes(k.value)}
              onChange={(v) => setReq({ ...req, kinds: v ? [...req.kinds, k.value] : req.kinds.filter((x) => x !== k.value) as WorkloadKind[] })} />
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend className="mb-1.5 text-sm text-ink-2">Schedulers</legend>
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          {SCHEDS.map((s) => (
            <Checkbox key={s} label={s === "rl" && !ckpts.length ? "RL agent (no checkpoint)" : schedLabel(s)}
              disabled={s === "rl" && !ckpts.length} checked={req.schedulers.includes(s)}
              onChange={(v) => setReq({ ...req, schedulers: v ? [...req.schedulers, s] : req.schedulers.filter((x) => x !== s) })} />
          ))}
        </div>
      </fieldset>
      {useRl && (
        <div className="grid max-w-xl grid-cols-2 gap-3">
          <Field label="Model">
            {(id) => <Select id={id} value={req.model_id || ckpts[0] || ""} onChange={(v) => setReq({ ...req, model_id: v })}
              options={ckpts.map((c) => ({ value: c, label: c }))} />}
          </Field>
          <Field label="Checkpoint">
            {(id) => <Select id={id} value={req.model_checkpoint ?? "best"} onChange={(v) => setReq({ ...req, model_checkpoint: v })}
              options={[{ value: "best", label: "Best on validation" }, { value: "final", label: "Final" }]} />}
          </Field>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" busy={busy} disabled={!req.kinds.length || !req.schedulers.length} onClick={run}>
          Run benchmark
        </Button>
        <span className="text-sm text-ink-3 tnum">{runs.toLocaleString()} simulations</span>
      </div>
      <ErrorNote error={error} />
    </div>
  );
}

function CompareTwo({ a, list }: { a: Experiment; list: ExperimentMeta[] }) {
  const [otherId, setOtherId] = useState("");
  const [loaded, setLoaded] = useState<Experiment | null>(null);
  useEffect(() => {
    if (otherId) api.experiment(otherId).then(setLoaded).catch(() => setLoaded(null));
  }, [otherId]);
  const b = loaded && loaded.id === otherId ? loaded : null;
  const metrics = ["avg_waiting", "avg_response", "max_wait", "jain_fairness", "objective"] as const;
  const rowsA = a.summary.groups.filter((g) => g.kind === "ALL");
  return (
    <div className="flex flex-col gap-3">
      <div className="max-w-xl">
        <Field label="Compare with">
          {(id) => <Select id={id} value={otherId} onChange={setOtherId}
            options={[{ value: "", label: "Choose an experiment" }, ...list.filter((e) => e.id !== a.id).map((e) => ({ value: e.id, label: e.id }))]} />}
        </Field>
      </div>
      {b && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm tnum">
            <thead>
              <tr className="text-left text-ink-3">
                <th className="py-2 pr-3 font-normal">Scheduler</th>
                {metrics.map((m) => <th key={m} className="py-2 pr-3 text-right font-normal">{m.replace("_", " ")}: this, other</th>)}
              </tr>
            </thead>
            <tbody>
              {rowsA.map((ga) => {
                const gb = b.summary.groups.find((g) => g.kind === "ALL" && g.scheduler === ga.scheduler);
                return (
                  <tr key={ga.scheduler} className="border-t border-rule">
                    <td className="py-1.5 pr-3">{schedLabel(ga.scheduler)}</td>
                    {metrics.map((m) => (
                      <td key={m} className="py-1.5 pr-3 text-right">
                        {num(ga[m].mean)}, <span className="text-ink-2">{gb ? num(gb[m].mean) : "–"}</span>
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-ink-3">Means over all traces. The experiments may use different workloads or seeds; check both configurations.</p>
        </div>
      )}
    </div>
  );
}

export function Experiments() {
  const [list, setList] = useState<ExperimentMeta[] | null>(null);
  const [current, setCurrent] = useState<Experiment | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [repro, setRepro] = useState<ReproduceResult | null>(null);
  const [reproBusy, setReproBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const refresh = useCallback(async () => {
    try {
      setList((await api.experiments()).experiments);
    } catch (e) {
      setError(e);
    }
  }, []);
  useEffect(() => {
    api.experiments().then((r) => setList(r.experiments)).catch(setError);
  }, []);

  const open = async (id: string) => {
    setLoadingId(id);
    setRepro(null);
    setError(null);
    try {
      setCurrent(await api.experiment(id));
    } catch (e) {
      setError(e);
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <Page title="Experiments" intro="Run reproducible benchmarks on held-out seeds, inspect saved results, and export the raw rows.">
      <Panel title="New benchmark">
        <RunForm onDone={(e) => { setCurrent(e); setRepro(null); void refresh(); }} />
      </Panel>
      <ErrorNote error={error} onRetry={refresh} />
      <Panel title="Saved experiments" pad={false}>
        {list === null ? <div className="px-5"><Loading /></div> : list.length === 0 ? (
          <div className="p-5"><Empty title="Nothing saved yet">Run a benchmark above; every run is saved with its seeds so it can be reproduced.</Empty></div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm tnum">
              <thead><tr className="text-left text-ink-3">
                <th className="px-5 py-2 font-normal">Experiment</th><th className="font-normal">Created</th>
                <th className="font-normal">Traces</th><th className="font-normal">Schedulers</th><th className="font-normal">Model</th>
              </tr></thead>
              <tbody>
                {list.map((e) => (
                  <tr key={e.id} className={`border-t border-rule ${current?.id === e.id ? "bg-panel-2" : ""}`}>
                    <td className="px-5 py-1.5">
                      <button type="button" className="text-left hover:underline" onClick={() => open(e.id)}>
                        {e.id}{loadingId === e.id ? "…" : ""}
                      </button>
                    </td>
                    <td className="text-ink-2">{new Date(e.created_at).toLocaleString()}</td>
                    <td>{e.config.kinds.length * e.config.seeds.length}</td>
                    <td className="text-ink-2">{e.config.schedulers.map(schedLabel).join(", ")}</td>
                    <td className="text-agent">{e.config.model_id ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {current && (
        <Panel title={current.id} aside={`${current.rows.length} result rows, ${num(current.duration_s)} s`}>
          <div className="flex flex-col gap-6">
            {experimentProvenance(current)}
            <div className="flex flex-wrap gap-2">
              <a className="inline-flex h-9 items-center rounded-md bg-panel-2 px-3.5 text-sm hover:bg-rule" href={api.exportUrl(current.id, "csv")} download>Download CSV</a>
              <a className="inline-flex h-9 items-center rounded-md bg-panel-2 px-3.5 text-sm hover:bg-rule" href={api.exportUrl(current.id, "json")} download>Download JSON</a>
              <Button busy={reproBusy} onClick={async () => {
                setReproBusy(true);
                try { setRepro(await api.reproduce(current.id)); } catch (e) { setError(e); } finally { setReproBusy(false); }
              }}>Reproduce from seeds</Button>
            </div>
            {repro && (
              <p role="status" className={`text-sm ${repro.identical ? "text-good" : "text-warn"}`}>
                {repro.identical
                  ? `✓ Reproduced: all ${repro.rows} rows match exactly.`
                  : `⚠ Re-run differs: largest difference ${repro.max_abs_difference}, ${repro.missing_rows} rows missing.`}
              </p>
            )}
            <SummaryTable exp={current} />
            {current.summary.head_to_head.length > 0 && (
              <div><h3 className="mb-2 font-semibold">Head to head</h3><HeadToHeadTable exp={current} /></div>
            )}
            <ExperimentCharts exp={current} />
            {list && list.length > 1 && (
              <div><h3 className="mb-2 font-semibold">Compare with another experiment</h3><CompareTwo a={current} list={list} /></div>
            )}
          </div>
        </Panel>
      )}
    </Page>
  );
}
