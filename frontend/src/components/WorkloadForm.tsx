import { useState } from "react";
import type { Distribution, WorkloadGenerate, WorkloadKind } from "../api/types";
import { KIND_LABELS } from "../lib/format";
import { Field, NumberInput, Select } from "./ui";

export const KIND_OPTIONS = (Object.keys(KIND_LABELS) as string[])
  .filter((k) => k !== "ALL")
  .map((k) => ({ value: k as WorkloadKind, label: KIND_LABELS[k]! }));

export const DEFAULT_WORKLOAD: WorkloadGenerate = { kind: "mixed", n_processes: 20, seed: 1 };

/** Seeded generator settings. Overrides stay empty (= preset) unless the user opens them. */
export function WorkloadForm({ value, onChange, maxProcesses = 500 }: {
  value: WorkloadGenerate;
  onChange: (w: WorkloadGenerate) => void;
  maxProcesses?: number;
}) {
  const [open, setOpen] = useState(
    value.mean_cpu_burst != null || value.mean_interarrival != null || value.io_fraction != null,
  );
  const set = <K extends keyof WorkloadGenerate>(k: K, v: WorkloadGenerate[K]) =>
    onChange({ ...value, [k]: v });
  const opt = (k: keyof WorkloadGenerate) => (v: number) =>
    set(k, Number.isFinite(v) ? v : null);
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label="Workload type">
          {(id) => <Select id={id} value={value.kind} onChange={(v) => set("kind", v)}
            options={KIND_OPTIONS} />}
        </Field>
        <Field label="Processes">
          {(id) => <NumberInput id={id} value={value.n_processes} min={1} max={maxProcesses}
            onChange={(v) => set("n_processes", v)} />}
        </Field>
        <Field label="Seed" hint="Same seed, same trace">
          {(id) => <NumberInput id={id} value={value.seed} min={0}
            onChange={(v) => set("seed", v)} />}
        </Field>
      </div>
      <button type="button" className="self-start text-sm text-ink-2 underline-offset-4 hover:underline"
        aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? "Hide distribution settings" : "Adjust arrival and burst distributions"}
      </button>
      {open && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Field label="Mean gap between arrivals" hint="Blank = preset">
            {(id) => <NumberInput id={id} value={value.mean_interarrival ?? NaN} min={0.1}
              step={0.5} onChange={opt("mean_interarrival")} />}
          </Field>
          <Field label="Mean CPU burst" hint="Blank = preset">
            {(id) => <NumberInput id={id} value={value.mean_cpu_burst ?? NaN} min={1}
              onChange={opt("mean_cpu_burst")} />}
          </Field>
          <Field label="Burst distribution">
            {(id) => (
              <Select id={id} value={(value.burst_distribution ?? "") as Distribution | ""}
                onChange={(v) => set("burst_distribution", v === "" ? null : v)}
                options={[
                  { value: "", label: "Preset" },
                  { value: "exponential", label: "Exponential" },
                  { value: "uniform", label: "Uniform" },
                  { value: "lognormal", label: "Log-normal" },
                ]} />
            )}
          </Field>
          <Field label="Share doing I/O" hint="0 to 1; blank = preset">
            {(id) => <NumberInput id={id} value={value.io_fraction ?? NaN} min={0} max={1}
              step={0.1} onChange={opt("io_fraction")} />}
          </Field>
          <Field label="Mean I/O burst">
            {(id) => <NumberInput id={id} value={value.mean_io_burst ?? NaN} min={1}
              onChange={opt("mean_io_burst")} />}
          </Field>
          <Field label="Priority levels">
            {(id) => <NumberInput id={id} value={value.priority_levels ?? 5} min={1} max={20}
              onChange={(v) => set("priority_levels", v)} />}
          </Field>
        </div>
      )}
    </div>
  );
}

/** Drop NaN/null overrides so the backend applies preset values. */
export function cleanWorkload(w: WorkloadGenerate): WorkloadGenerate {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(w)) {
    if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) continue;
    out[k] = v;
  }
  return out as unknown as WorkloadGenerate;
}
