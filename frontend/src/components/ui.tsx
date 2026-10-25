import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useId } from "react";
import type { ModelState } from "../api/types";

export function Page({ title, intro, actions, children }: {
  title: string;
  intro?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto w-full max-w-[1240px] px-4 pb-16 pt-8 md:px-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[1.9rem] font-semibold">{title}</h1>
          {intro && <p className="mt-1.5 max-w-[68ch] text-ink-2">{intro}</p>}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </header>
      <div className="flex flex-col gap-6">{children}</div>
    </div>
  );
}

export function Panel({ title, aside, children, className = "", pad = true }: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
}) {
  return (
    <section className={`min-w-0 rounded-lg bg-panel ${className}`}>
      {(title || aside) && (
        <div className="flex flex-wrap items-baseline justify-between gap-2 px-5 pt-4">
          {title && <h2 className="text-[1.05rem] font-semibold">{title}</h2>}
          {aside && <div className="text-sm text-ink-3">{aside}</div>}
        </div>
      )}
      <div className={pad ? "p-5" : ""}>{children}</div>
    </section>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger" | "agent";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-run text-white hover:brightness-110",
  agent: "bg-agent text-[#14101f] hover:brightness-110",
  secondary: "bg-panel-2 text-ink hover:bg-rule",
  ghost: "text-ink-2 hover:bg-panel-2 hover:text-ink",
  danger: "bg-panel-2 text-bad hover:bg-rule",
};

export function Button({ variant = "secondary", busy, children, className = "", ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; busy?: boolean }) {
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || busy}
      aria-busy={busy || undefined}
      className={`inline-flex h-9 items-center justify-center gap-2 rounded-md px-3.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${className}`}
    >
      {busy && <Spinner />}
      {children}
    </button>
  );
}

export function Spinner() {
  return (
    <span
      aria-hidden
      className="inline-block size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
    />
  );
}

export function Field({ label, hint, children }: {
  label: string;
  hint?: string;
  children: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={id} className="text-sm text-ink-2">
        {label}
      </label>
      {children(id)}
      {hint && <span className="text-xs text-ink-3">{hint}</span>}
    </div>
  );
}

const INPUT =
  "h-9 w-full rounded-md border border-rule bg-ground px-2.5 text-sm text-ink tnum outline-none focus-visible:border-[var(--focus)]";

export function NumberInput({ id, value, onChange, min, max, step = 1, disabled }: {
  id?: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <input
      id={id}
      type="number"
      className={INPUT}
      value={Number.isFinite(value) ? value : ""}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
    />
  );
}

export function TextInput({ id, value, onChange, placeholder }: {
  id?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input id={id} className={INPUT} value={value} placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)} />
  );
}

export function Select<T extends string>({ id, value, onChange, options, disabled }: {
  id?: string;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; disabled?: boolean }[];
  disabled?: boolean;
}) {
  return (
    <select id={id} className={INPUT} value={value} disabled={disabled}
      onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Checkbox({ checked, onChange, label, disabled }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={`inline-flex items-center gap-2 text-sm ${disabled ? "opacity-50" : ""}`}>
      <input type="checkbox" className="size-4 accent-[var(--run)]" checked={checked}
        disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

export function ErrorNote({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="flex flex-wrap items-start justify-between gap-3 rounded-md border border-bad/40 bg-bad/10 px-4 py-3 text-sm">
      <span><strong className="font-semibold text-bad">Request failed.</strong> {msg}</span>
      {onRetry && <Button variant="ghost" onClick={onRetry}>Try again</Button>}
    </div>
  );
}

export function Empty({ title, children, action }: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-md border border-dashed border-rule px-5 py-8">
      <p className="font-medium">{title}</p>
      {children && <div className="max-w-[60ch] text-sm text-ink-2">{children}</div>}
      {action}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-6 text-sm text-ink-3">
      <Spinner /> {label}…
    </div>
  );
}

const MODEL_STATE: Record<ModelState, { label: string; color: string; glyph: string }> = {
  loaded: { label: "Trained model loaded", color: "var(--good)", glyph: "●" },
  training: { label: "Model training", color: "var(--warn)", glyph: "◐" },
  unavailable: { label: "No model loaded", color: "var(--ink-3)", glyph: "○" },
  dependencies_missing: { label: "RL packages missing", color: "var(--bad)", glyph: "✕" },
};

/** Model state always pairs a glyph and words with the color. */
export function ModelBadge({ state, detail }: { state: ModelState; detail?: string }) {
  const s = MODEL_STATE[state];
  return (
    <span className="inline-flex items-center gap-1.5 text-sm" data-testid="model-badge">
      <span aria-hidden style={{ color: s.color }}>{s.glyph}</span>
      <span>{s.label}</span>
      {detail && <span className="text-ink-3">{detail}</span>}
    </span>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-sm text-ink-3">{label}</div>
      <div className="tnum text-[1.6rem] font-medium leading-tight" style={{ fontStretch: "105%" }}>
        {value}
      </div>
      {sub && <div className="text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

export function Provenance({ children }: { children: ReactNode }) {
  return (
    <p className="text-xs text-ink-3">
      <span className="mr-1.5 rounded-sm bg-run/15 px-1.5 py-0.5 text-run">Measured</span>
      {children}
    </p>
  );
}
