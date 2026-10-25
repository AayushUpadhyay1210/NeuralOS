import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { num } from "../lib/format";

const AXIS = { stroke: "var(--rule)", tick: { fill: "var(--ink-3)", fontSize: 11 }, tickLine: false };

export interface BarDatum {
  key: string;
  label: string;
  value: number;
  err?: number;
  isAgent?: boolean;
}

function TipBox({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="rounded-md bg-panel-2 px-2.5 py-1.5 text-xs shadow-lg tnum">
      <div className="font-medium text-ink">{title}</div>
      {lines.map((l) => <div key={l} className="text-ink-2">{l}</div>)}
    </div>
  );
}

/**
 * One metric across schedulers. Single series, so bars share one hue; the RL
 * agent keeps its reserved accent so it reads the same on every chart.
 */
export function MetricBars({ title, unit, data, lowerIsBetter, format = (x) => num(x), neutral }: {
  title: string;
  unit?: string;
  data: BarDatum[];
  lowerIsBetter: boolean;
  format?: (x: number) => string;
  /** Descriptive data with no better/worse direction. */
  neutral?: boolean;
}) {
  // Many categories read better as horizontal bars: labels never collide.
  const horizontal = data.length > 6;
  const axisFmt = (v: number) => {
    const t = format(v);
    return t.endsWith("%") ? `${Math.round(v * 100)}%` : t;
  };
  const [table, setTable] = useState(false);
  const best = data.length
    ? data.reduce((a, b) => ((lowerIsBetter ? b.value < a.value : b.value > a.value) ? b : a))
    : null;
  return (
    <figure className="flex min-w-0 flex-col gap-1">
      <figcaption className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">{title}</span>
        <button type="button" className="text-xs text-ink-3 hover:text-ink"
          onClick={() => setTable(!table)} aria-pressed={table}>
          {table ? "Chart" : "Table"}
        </button>
      </figcaption>
      <div className="text-xs text-ink-3">
        {neutral ? unit : <>{unit ? `${unit}, ` : ""}{lowerIsBetter ? "lower is better" : "higher is better"}</>}
        {data.some((d) => d.err) ? "; whiskers show 95% CI" : ""}
      </div>
      {table ? (
        <table className="mt-1 text-sm tnum">
          <tbody>
            {data.map((d) => (
              <tr key={d.key} className="border-t border-rule">
                <td className="py-1 pr-3">{d.label}</td>
                <td className="py-1 text-right">{format(d.value)}{d.err ? ` ± ${format(d.err)}` : ""}</td>
                <td className="py-1 pl-2 text-xs text-ink-3">{!neutral && best?.key === d.key ? "best" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={{ height: horizontal ? data.length * 26 + 36 : 208 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"}
              margin={{ top: 8, right: 8, left: horizontal ? 0 : -8, bottom: 0 }} barCategoryGap="28%">
              <CartesianGrid vertical={horizontal} horizontal={!horizontal} stroke="var(--rule)" strokeWidth={0.6} />
              {horizontal ? (
                <>
                  <XAxis type="number" {...AXIS} tickFormatter={axisFmt} />
                  <YAxis type="category" dataKey="label" {...AXIS} width={104} interval={0}
                    tick={{ fill: "var(--ink-2)", fontSize: 11 }} />
                </>
              ) : (
                <>
                  <XAxis dataKey="label" {...AXIS} interval={0} height={34}
                    tick={{ fill: "var(--ink-3)", fontSize: 10 }} />
                  <YAxis {...AXIS} axisLine={false} width={50} tickFormatter={axisFmt} />
                </>
              )}
              <Tooltip cursor={{ fill: "var(--panel-2)", opacity: 0.6 }}
                content={({ active, payload }) => {
                  const d = active && payload?.[0] ? (payload[0].payload as BarDatum) : null;
                  if (!d) return null;
                  return <TipBox title={d.label}
                    lines={[`${format(d.value)}${d.err ? ` ± ${format(d.err)} (95% CI)` : ""}${unit ? ` ${unit}` : ""}`]} />;
                }} />
              <Bar dataKey="value" radius={horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]} isAnimationActive={false}>
                {data.map((d) => (
                  <Cell key={d.key} fill={d.isAgent ? "var(--agent)" : "var(--run)"} />
                ))}
                {data.some((d) => d.err) && (
                  <ErrorBar dataKey="err" width={6} stroke="var(--ink-2)" strokeWidth={1.2}
                    direction={horizontal ? "x" : "y"} />
                )}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  );
}

export interface RefLine {
  label: string;
  value: number;
}

/** A single measured series over training steps, with labeled reference levels. */
export function SeriesChart({ title, data, xKey, yKey, yLabel, refs = [], color = "var(--agent)",
  lowerIsBetter }: {
  title: string;
  data: Record<string, number>[];
  xKey: string;
  yKey: string;
  yLabel: string;
  refs?: RefLine[];
  color?: string;
  lowerIsBetter?: boolean;
}) {
  const [table, setTable] = useState(false);
  return (
    <figure className="flex min-w-0 flex-col gap-1">
      <figcaption className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">{title}</span>
        <button type="button" className="text-xs text-ink-3 hover:text-ink"
          onClick={() => setTable(!table)} aria-pressed={table}>
          {table ? "Chart" : "Table"}
        </button>
      </figcaption>
      <div className="text-xs text-ink-3">
        {yLabel}{lowerIsBetter === undefined ? "" : lowerIsBetter ? ", lower is better" : ", higher is better"}
        {refs.length ? "; dashed lines are baselines on the same validation set" : ""}
      </div>
      {table ? (
        <div className="max-h-60 overflow-auto">
          <table className="mt-1 w-full text-sm tnum">
            <tbody>
              {data.map((d, i) => (
                <tr key={i} className="border-t border-rule">
                  <td className="py-1 pr-3 text-ink-3">{num(d[xKey])} steps</td>
                  <td className="py-1 text-right">{num(d[yKey], 3)}</td>
                </tr>
              ))}
              {refs.map((r) => (
                <tr key={r.label} className="border-t border-rule text-ink-3">
                  <td className="py-1 pr-3">{r.label}</td>
                  <td className="py-1 text-right">{num(r.value, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 64, left: -4, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={0.6} />
              <XAxis dataKey={xKey} type="number" domain={["dataMin", "dataMax"]} {...AXIS}
                tickFormatter={(v: number) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v))} />
              <YAxis {...AXIS} axisLine={false} width={48} domain={["auto", "auto"]}
                tickFormatter={(v: number) => num(v, 1)} />
              <Tooltip cursor={{ stroke: "var(--ink-3)", strokeWidth: 1 }}
                content={({ active, payload }) => {
                  const d = active && payload?.[0] ? (payload[0].payload as Record<string, number>) : null;
                  if (!d) return null;
                  return <TipBox title={`${num(d[xKey])} steps`} lines={[`${yLabel}: ${num(d[yKey], 3)}`]} />;
                }} />
              {refs.map((r) => (
                <ReferenceLine key={r.label} y={r.value} stroke="var(--ink-3)" strokeDasharray="4 4"
                  label={{ value: r.label, position: "right", fill: "var(--ink-2)", fontSize: 10 }} />
              ))}
              <Line type="linear" dataKey={yKey} stroke={color} strokeWidth={2} isAnimationActive={false}
                dot={data.length < 40 ? { r: 3, fill: color, stroke: "var(--panel)", strokeWidth: 2 } : false}
                activeDot={{ r: 5, stroke: "var(--panel)", strokeWidth: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  );
}
