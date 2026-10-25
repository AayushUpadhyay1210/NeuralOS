import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Experiment, ModelMeta, RewardWeights, TrainingStatus } from "../api/types";
import { SeriesChart } from "../components/Charts";
import { experimentProvenance, HeadToHeadTable, SummaryTable } from "../components/ExperimentView";
import { Button, Empty, ErrorNote, Field, Loading, ModelBadge, NumberInput, Page, Panel, TextInput } from "../components/ui";
import { duration, num, schedLabel } from "../lib/format";
import { useApp } from "../state/AppState";

const STATE_TEXT: Record<string, string> = {
  starting: "Starting (loading PyTorch)",
  running: "Training",
  completed: "Completed",
  cancelled: "Cancelled",
  failed: "Failed",
  interrupted: "Interrupted (the API restarted during training)",
  unknown: "Unknown",
};

function StartForm({ onStarted, defaults, disabled }: {
  onStarted: (id: string) => void;
  defaults: RewardWeights | undefined;
  disabled: string | null;
}) {
  const [runId, setRunId] = useState("");
  const [steps, setSteps] = useState(100_000);
  const [seed, setSeed] = useState(0);
  const [reward, setReward] = useState<RewardWeights>(defaults ?? { wait: 1, fair: 0.1, resp: 0.25, switch: 0.1 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.startTraining({ run_id: runId || undefined, timesteps: steps, seed, reward });
      onStarted(r.run_id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  const w = (k: keyof RewardWeights, label: string, hint: string) => (
    <Field label={label} hint={hint}>
      {(id) => <NumberInput id={id} value={reward[k]} min={0} step={0.05}
        onChange={(v) => setReward({ ...reward, [k]: v })} />}
    </Field>
  );
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label="Run name" hint="Optional; letters, digits, - and _">
          {(id) => <TextInput id={id} value={runId} onChange={setRunId} placeholder="ppo-my-run" />}
        </Field>
        <Field label="Training steps" hint={`About ${duration(steps / 800)} on a 2-core CPU`}>
          {(id) => <NumberInput id={id} value={steps} min={1000} max={1_000_000} step={10_000} onChange={setSteps} />}
        </Field>
        <Field label="Seed">
          {(id) => <NumberInput id={id} value={seed} min={0} onChange={setSeed} />}
        </Field>
      </div>
      <div>
        <h3 className="mb-2 text-sm font-semibold">Reward weights</h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {w("wait", "Waiting", "Per tick a process waits")}
          {w("fair", "Long-wait penalty", "Quadratic in each wait; deters starvation")}
          {w("resp", "Response", "Per tick before first run")}
          {w("switch", "Context switch", "Per switch, in mean bursts")}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="agent" busy={busy} disabled={!!disabled} onClick={start}>Start training</Button>
        {disabled && <span className="text-sm text-ink-3">{disabled}</span>}
      </div>
      <ErrorNote error={error} />
    </div>
  );
}

async function fetchRun(runId: string): Promise<{ st: TrainingStatus; meta: ModelMeta | null }> {
  const st = await api.trainingStatus(runId);
  let meta: ModelMeta | null = null;
  try {
    meta = (await api.model(runId)).metadata;
  } catch {
    /* metadata appears a few seconds after a run starts */
  }
  return { st, meta };
}

function RunDetail({ runId, onChanged }: { runId: string; onChanged: () => void }) {
  const { modelStatus, refreshModelStatus } = useApp();
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [evalRes, setEvalRes] = useState<Experiment | null>(null);

  const load = useCallback(
    () => fetchRun(runId).then(({ st, meta: m }) => {
      setStatus(st);
      setMeta(m);
      setError(null);
    }).catch(setError),
    [runId],
  );

  useEffect(() => {
    fetchRun(runId).then(({ st, meta: m }) => {
      setStatus(st);
      setMeta(m);
    }).catch(setError);
  }, [runId]);
  const live = status?.state === "running" || status?.state === "starting";
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(load, 2000);
    return () => window.clearInterval(id);
  }, [live, load]);
  useEffect(() => {
    if (!live) onChanged();
  }, [live, onChanged]);

  const act = async (what: string, fn: () => Promise<unknown>) => {
    setBusy(what);
    setError(null);
    try {
      await fn();
      await refreshModelStatus();
      await load();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(null);
    }
  };

  const p = status?.progress;
  const isActive = modelStatus?.active?.run_id === runId;
  const baselineRefs = p ? Object.entries(p.baselines)
    .filter(([k]) => ["sjf", "rr", "fcfs"].includes(k))
    .map(([k, v]) => ({ label: schedLabel(k), value: v.objective })) : [];
  const frac = p ? Math.min(1, p.timesteps / p.total_timesteps) : 0;

  return (
    <Panel title={runId} aside={status && (
      <span className={status.state === "failed" ? "text-bad" : status.state === "running" ? "text-warn" : ""}>
        {STATE_TEXT[status.state] ?? status.state}
      </span>
    )}>
      <div className="flex flex-col gap-5">
        <ErrorNote error={error} />
        {status?.error && <ErrorNote error={status.error} />}
        {status?.detail && <ErrorNote error={status.detail} />}
        {p && (
          <div>
            <div className="mb-1 flex justify-between text-sm tnum text-ink-2">
              <span>{p.timesteps.toLocaleString()} of {p.total_timesteps.toLocaleString()} steps</span>
              <span>{duration(p.elapsed_s)} elapsed, {Math.round(p.fps)} steps/s</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-panel-2" role="progressbar"
              aria-valuenow={Math.round(frac * 100)} aria-valuemin={0} aria-valuemax={100}
              aria-label="Training progress">
              <div className="h-full bg-agent transition-[width]" style={{ width: `${frac * 100}%` }} />
            </div>
          </div>
        )}
        {!p && live && <Loading label="Waiting for the first progress report" />}

        {meta && (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
            <div><dt className="text-ink-3">Algorithm</dt><dd>{meta.algorithm} (action-masked PPO)</dd></div>
            <div><dt className="text-ink-3">Seed</dt><dd className="tnum">{meta.seed}</dd></div>
            <div><dt className="text-ink-3">Steps done</dt><dd className="tnum">{meta.timesteps_done.toLocaleString()}</dd></div>
            <div><dt className="text-ink-3">Checkpoints</dt>
              <dd>{[meta.has_best || p?.best ? "best" : null, meta.has_final ? "final" : null].filter(Boolean).join(", ") || "none yet"}</dd></div>
            <div><dt className="text-ink-3">Reward weights</dt>
              <dd className="tnum">wait {meta.reward.wait}, long-wait {meta.reward.fair}, response {meta.reward.resp}, switch {meta.reward.switch}</dd></div>
            <div><dt className="text-ink-3">Action space</dt>
              <dd className="tnum">{meta.space.slots} slots × quanta {meta.space.quanta.map((q) => q ?? "full burst").join(" / ")}</dd></div>
            <div><dt className="text-ink-3">Best validation</dt>
              <dd className="tnum">{meta.best ? `${num(meta.best.objective, 2)} at ${meta.best.timesteps?.toLocaleString()} steps` : "–"}</dd></div>
            <div><dt className="text-ink-3">Versions</dt>
              <dd className="text-ink-2">torch {meta.versions.torch}, SB3 {meta.versions.stable_baselines3}</dd></div>
          </dl>
        )}

        {p && p.curve.length > 1 && (
          <div className="grid gap-8 lg:grid-cols-2">
            <SeriesChart title="Training reward" data={p.curve} xKey="timesteps" yKey="ep_reward_mean"
              yLabel="Mean episode return (negative cost)" lowerIsBetter={false} />
            {p.validation.length > 0 && (
              <SeriesChart title="Validation objective" lowerIsBetter
                data={p.validation.map((v) => ({ timesteps: v.timesteps ?? 0, objective: v.objective }))}
                xKey="timesteps" yKey="objective" yLabel="Episode cost on held-out validation workloads"
                refs={baselineRefs} />
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          {live && (
            <Button variant="danger" busy={busy === "cancel"} onClick={() => act("cancel", () => api.cancelTraining(runId))}>
              Cancel training
            </Button>
          )}
          {(meta?.has_best || meta?.has_final) && (
            <>
              <Button variant="agent" busy={busy === "load"} disabled={isActive && modelStatus?.active?.checkpoint === "best"}
                onClick={() => act("load", () => api.loadModel(runId, meta?.has_best ? "best" : "final"))}>
                {isActive ? "Loaded" : "Load best checkpoint"}
              </Button>
              {meta?.has_final && (
                <Button busy={busy === "loadf"} onClick={() => act("loadf", () => api.loadModel(runId, "final"))}>
                  Load final checkpoint
                </Button>
              )}
              <Button busy={busy === "eval"} onClick={() => act("eval", async () => {
                setEvalRes(await api.evaluateModel(runId, { checkpoint: meta?.has_best ? "best" : "final", n_seeds: 5, n_processes: 30 }));
              })}>Evaluate on test seeds</Button>
            </>
          )}
          {isActive && <Button variant="ghost" onClick={() => act("unload", api.unloadModel)}>Unload</Button>}
        </div>
        {busy === "eval" && <Loading label="Running every scheduler on 35 held-out traces" />}
        {evalRes && (
          <div className="flex flex-col gap-3 border-t border-rule pt-4">
            <h3 className="font-semibold">Evaluation on held-out test seeds</h3>
            {experimentProvenance(evalRes)}
            <SummaryTable exp={evalRes} />
            <HeadToHeadTable exp={evalRes} />
            <p className="text-xs text-ink-3">Saved as experiment {evalRes.id}.</p>
          </div>
        )}
      </div>
    </Panel>
  );
}

export function Training() {
  const { config, modelStatus, health } = useApp();
  const [models, setModels] = useState<ModelMeta[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const refresh = useCallback(
    () => api.models().then((r) => {
      setModels(r.models);
      setSelected((cur) => cur ?? r.models[0]?.run_id ?? null);
    }).catch(setError),
    [],
  );
  useEffect(() => {
    api.models().then((r) => {
      setModels(r.models);
      setSelected((cur) => cur ?? r.models[0]?.run_id ?? null);
    }).catch(setError);
  }, []);

  const missing = health?.missing_dependencies ?? [];
  const running = modelStatus?.training ?? [];
  const disabled = missing.length ? `Install the RL packages first: ${missing.join(", ")}`
    : running.length ? `${running[0]} is already training` : null;

  return (
    <Page title="AI training"
      intro="Train an action-masked PPO policy on seeded training workloads, watch validation against the baselines, and load a checkpoint for inference."
      actions={modelStatus && <ModelBadge state={modelStatus.state}
        detail={modelStatus.active ? `${modelStatus.active.run_id} (${modelStatus.active.checkpoint})` : undefined} />}>
      {missing.length > 0 && (
        <ErrorNote error={`RL dependencies are missing on the API server: ${missing.join(", ")}. Run: pip install -r backend/requirements-rl.txt`} />
      )}
      <Panel title="Start a training run">
        <StartForm defaults={config?.reward_defaults} disabled={disabled}
          onStarted={(id) => { setSelected(id); void refresh(); }} />
      </Panel>
      <ErrorNote error={error} onRetry={refresh} />
      {models === null ? <Loading label="Loading models" /> : models.length === 0 && !selected ? (
        <Empty title="No trained models yet">
          Start a run above, or train from a terminal with <code>python -m neuralos.rl.train</code>. Until a checkpoint exists, the RL agent is unavailable everywhere in the app.
        </Empty>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
          <Panel title="Runs" pad={false}>
            <ul className="flex flex-col py-2">
              {(models ?? []).map((m) => (
                <li key={m.run_id}>
                  <button type="button" onClick={() => setSelected(m.run_id)} aria-current={selected === m.run_id}
                    className={`flex w-full flex-col items-start px-5 py-2 text-left ${selected === m.run_id ? "bg-panel-2" : "hover:bg-panel-2/60"}`}>
                    <span className={modelStatus?.active?.run_id === m.run_id ? "text-agent" : ""}>{m.run_id}</span>
                    <span className="text-xs text-ink-3 tnum">
                      {STATE_TEXT[m.status] ?? m.status}, {m.timesteps_done.toLocaleString()} steps
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
          {selected && <RunDetail key={selected} runId={selected} onChanged={refresh} />}
        </div>
      )}
    </Page>
  );
}
