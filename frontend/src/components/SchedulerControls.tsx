import type { SchedulerIn, SchedulerName } from "../api/types";
import { SCHEDULER_LABELS } from "../lib/format";
import { useApp } from "../state/AppState";
import { Field, NumberInput, Select } from "./ui";

export function SchedulerControls({ value, onChange, contextSwitch, onContextSwitch }: {
  value: SchedulerIn;
  onChange: (s: SchedulerIn) => void;
  contextSwitch: number;
  onContextSwitch: (v: number) => void;
}) {
  const { modelStatus } = useApp();
  const rlReady = modelStatus?.state === "loaded";
  const options = (Object.keys(SCHEDULER_LABELS) as SchedulerName[]).map((n) => ({
    value: n,
    label: n === "rl" && !rlReady ? "RL agent (load a model first)" : SCHEDULER_LABELS[n]!,
    disabled: n === "rl" && !rlReady,
  }));
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Field label="Scheduler">
        {(id) => <Select id={id} value={value.name} options={options}
          onChange={(name) => onChange({ ...value, name })} />}
      </Field>
      <Field label="Round Robin quantum" hint={value.name === "rr" ? undefined : "Used by Round Robin"}>
        {(id) => <NumberInput id={id} value={value.quantum ?? 4} min={1} max={100}
          disabled={value.name !== "rr"} onChange={(v) => onChange({ ...value, quantum: v })} />}
      </Field>
      <Field label="Priority aging" hint="Ticks per priority step; blank = off">
        {(id) => <NumberInput id={id} value={value.aging ?? NaN} min={1}
          disabled={!value.name.startsWith("priority")}
          onChange={(v) => onChange({ ...value, aging: Number.isFinite(v) ? v : null })} />}
      </Field>
      <Field label="Context switch cost" hint="Ticks per switch">
        {(id) => <NumberInput id={id} value={contextSwitch} min={0} max={20}
          onChange={onContextSwitch} />}
      </Field>
      {value.name === "rl" && modelStatus?.active && (
        <p className="col-span-full text-sm text-ink-2">
          Using checkpoint <span className="text-agent">{modelStatus.active.run_id}</span> ({modelStatus.active.checkpoint}),
          trained {modelStatus.active.timesteps?.toLocaleString()} steps.
        </p>
      )}
    </div>
  );
}

export function schedulerPayload(s: SchedulerIn): SchedulerIn {
  const out: SchedulerIn = { name: s.name };
  if (s.name === "rr") out.quantum = s.quantum ?? 4;
  if (s.name.startsWith("priority") && s.aging) out.aging = s.aging;
  return out;
}
