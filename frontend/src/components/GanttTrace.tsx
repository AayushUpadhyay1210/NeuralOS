import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import type { Segment, TransitionOut } from "../api/types";
import { stateSpans } from "../lib/trace";

const LABEL_W = 52;
const CPU_H = 30;
const AXIS_H = 22;
const ROW_H = 20;
const BAR_H = 12;

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setW(Math.max(320, el.clientWidth));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function niceTicks(a: number, b: number, target = 8): number[] {
  const span = Math.max(1, b - a);
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const st = Math.max(1, Math.round(step));
  const out: number[] = [];
  for (let t = Math.ceil(a / st) * st; t <= b; t += st) out.push(t);
  return out;
}

export interface GanttProps {
  timeline: Segment[];
  transitions: TransitionOut[];
  pids: number[];
  cursor: number;
  onCursor: (t: number) => void;
  selection: [number, number] | null;
  onSelect: (sel: [number, number] | null) => void;
  view?: [number, number];
  highlightPid?: number | null;
  onPickPid?: (pid: number) => void;
  onStep?: (dir: 1 | -1) => void;
  maxHeight?: number;
}

interface Hover {
  x: number;
  y: number;
  text: string;
}

const STATE_WORD: Record<string, string> = {
  READY: "waiting in ready queue",
  RUNNING: "running",
  BLOCKED: "blocked on I/O",
};

export function GanttTrace(props: GanttProps) {
  const { timeline, transitions, pids, cursor, onCursor, selection, onSelect, highlightPid,
    onPickPid, onStep, maxHeight = 380 } = props;
  const end = timeline.length ? timeline[timeline.length - 1]!.end : 1;
  const [v0, v1] = props.view ?? [0, end];
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const plotW = width - LABEL_W - 8;
  const spans = useMemo(() => stateSpans(transitions), [transitions]);
  const x = (t: number) => LABEL_W + ((t - v0) / Math.max(1e-9, v1 - v0)) * plotW;
  const tAt = (px: number) =>
    Math.round(Math.min(v1, Math.max(v0, v0 + ((px - LABEL_W) / plotW) * (v1 - v0))));
  const ticks = niceTicks(v0, v1);
  const drag = useRef<{ x0: number; t0: number } | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const visible = (s: { start: number; end: number }) => s.end > v0 && s.start < v1;
  const clipW = (s: number, e: number) => Math.max(0, x(Math.min(e, v1)) - x(Math.max(s, v0)));

  const localX = (e: PointerEvent<SVGSVGElement>) =>
    e.clientX - e.currentTarget.getBoundingClientRect().left;

  const onDown = (e: PointerEvent<SVGSVGElement>) => {
    const px = localX(e);
    if (px < LABEL_W) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x0: px, t0: tAt(px) };
  };
  const onMove = (e: PointerEvent<SVGSVGElement>, lane: "cpu" | "rows") => {
    const px = localX(e);
    const py = e.clientY - e.currentTarget.getBoundingClientRect().top;
    if (drag.current && Math.abs(px - drag.current.x0) > 3) {
      const a = drag.current.t0;
      const b = tAt(px);
      if (a !== b) onSelect([Math.min(a, b), Math.max(a, b)]);
    }
    if (px < LABEL_W) {
      setHover(null);
      return;
    }
    const t = v0 + ((px - LABEL_W) / plotW) * (v1 - v0);
    let text = `t = ${t.toFixed(1)}`;
    if (lane === "cpu") {
      const seg = timeline.find((s) => s.start <= t && t < s.end);
      if (seg) {
        text = seg.kind === "run" ? `P${seg.pid} running, ${seg.start}–${seg.end}`
          : seg.kind === "switch" ? `Context switch to P${seg.pid}, ${seg.start}–${seg.end}`
          : `CPU idle, ${seg.start}–${seg.end}`;
      }
    } else {
      const pid = pids[Math.floor(py / ROW_H)];
      if (pid !== undefined) {
        const sp = (spans.get(pid) ?? []).find((s) => s.start <= t && t < s.end);
        text = sp ? `P${pid} ${STATE_WORD[sp.state] ?? sp.state}, ${sp.start}–${sp.end}`
          : `P${pid} not in system at t=${Math.floor(t)}`;
      }
    }
    setHover({ x: e.clientX, y: e.clientY, text });
  };
  const onUp = (e: PointerEvent<SVGSVGElement>, lane: "cpu" | "rows") => {
    const px = localX(e);
    const d = drag.current;
    drag.current = null;
    if (!d || Math.abs(px - d.x0) > 3) return;
    onCursor(tAt(px));
    if (lane === "rows" && onPickPid) {
      const py = e.clientY - e.currentTarget.getBoundingClientRect().top;
      const pid = pids[Math.floor(py / ROW_H)];
      if (pid !== undefined) onPickPid(pid);
    }
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight") onStep?.(1);
    else if (e.key === "ArrowLeft") onStep?.(-1);
    else if (e.key === "Escape") onSelect(null);
    else return;
    e.preventDefault();
  };

  const overlays = (h: number) => (
    <>
      {selection && visible({ start: selection[0], end: selection[1] }) &&
        !(selection[0] === v0 && selection[1] === v1) && (
        <rect x={x(Math.max(selection[0], v0))} y={0} width={clipW(selection[0], selection[1])}
          height={h} fill="var(--run-soft)" stroke="var(--run)" strokeWidth={1} />
      )}
      {cursor >= v0 && cursor <= v1 && (
        <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={h} stroke="var(--ink)" strokeWidth={1.5} />
      )}
    </>
  );

  const rowsH = pids.length * ROW_H;
  return (
    <div ref={wrapRef} data-testid="gantt" className="relative select-none" tabIndex={0} onKeyDown={onKey}
      role="group"
      aria-label={`Gantt trace from t=${v0} to t=${v1}. Click to place the cursor, drag to select an interval, arrow keys step between events.`}>
      <svg width={width} height={CPU_H + AXIS_H} className="block touch-none"
        onPointerDown={onDown} onPointerMove={(e) => onMove(e, "cpu")}
        onPointerUp={(e) => onUp(e, "cpu")} onPointerLeave={() => setHover(null)}>
        <defs>
          <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
          </pattern>
          <pattern id="hatch-io" width="4" height="4" patternUnits="userSpaceOnUse"
            patternTransform="rotate(135)">
            <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-2)" strokeWidth="1" />
          </pattern>
        </defs>
        <text x={0} y={19} fontSize={12} fill="var(--ink-2)">CPU</text>
        {timeline.filter(visible).map((s, i) => {
          const w = clipW(s.start, s.end);
          const fill = s.kind === "run" ? "var(--run)"
            : s.kind === "switch" ? "var(--switch)" : "url(#hatch)";
          return (
            <g key={i}>
              <rect x={x(Math.max(s.start, v0))} y={4} width={Math.max(0.5, w - (w > 3 ? 1 : 0))}
                height={22} fill={fill} opacity={s.kind === "idle" ? 0.5 : 1} rx={w > 6 ? 2 : 0}
                stroke={s.kind === "run" && s.pid === highlightPid ? "var(--ink)" : "none"} strokeWidth={1.5} />
              {s.kind === "run" && w > 24 && (
                <text x={x(Math.max(s.start, v0)) + 4} y={19} fontSize={11} fill="#fff"
                  className="tnum pointer-events-none">P{s.pid}</text>
              )}
            </g>
          );
        })}
        <line x1={LABEL_W} x2={LABEL_W + plotW} y1={CPU_H + 2} y2={CPU_H + 2} stroke="var(--rule)" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={CPU_H + 2} y2={CPU_H + 6} stroke="var(--ink-3)" />
            <text x={x(t)} y={CPU_H + 18} fontSize={11} textAnchor="middle" fill="var(--ink-3)"
              className="tnum">{t}</text>
          </g>
        ))}
        {overlays(CPU_H)}
      </svg>
      <div className="overflow-y-auto" style={{ maxHeight }}>
        <svg width={width} height={rowsH} className="block touch-none"
          onPointerDown={onDown} onPointerMove={(e) => onMove(e, "rows")}
          onPointerUp={(e) => onUp(e, "rows")} onPointerLeave={() => setHover(null)}>
          {ticks.map((t) => (
            <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={rowsH} stroke="var(--rule)"
              strokeWidth={0.5} />
          ))}
          {pids.map((pid, r) => {
            const y = r * ROW_H;
            const hl = pid === highlightPid;
            return (
              <g key={pid}>
                {hl && <rect x={0} y={y} width={width} height={ROW_H} fill="var(--panel-2)" />}
                <text x={4} y={y + 14} fontSize={11} className="tnum"
                  fill={hl ? "var(--ink)" : "var(--ink-2)"}>P{pid}</text>
                {(spans.get(pid) ?? []).filter(visible).map((s, i) => {
                  const w = clipW(s.start, s.end);
                  const yy = y + (ROW_H - BAR_H) / 2;
                  if (s.state === "RUNNING")
                    return <rect key={i} x={x(Math.max(s.start, v0))} y={yy} width={Math.max(0.75, w)}
                      height={BAR_H} fill="var(--run)" rx={w > 6 ? 2 : 0}
                      stroke={hl ? "var(--ink)" : "none"} strokeWidth={1.5} />;
                  if (s.state === "READY")
                    return <rect key={i} x={x(Math.max(s.start, v0))} y={yy + 3} width={w}
                      height={BAR_H - 6} fill="var(--wait)" />;
                  if (s.state === "BLOCKED")
                    return <rect key={i} x={x(Math.max(s.start, v0))} y={yy + 2} width={w}
                      height={BAR_H - 4} fill="url(#hatch-io)" />;
                  return null;
                })}
              </g>
            );
          })}
          {overlays(rowsH)}
        </svg>
      </div>
      {hover && (
        <div role="tooltip"
          className="pointer-events-none fixed z-50 rounded-md bg-panel-2 px-2.5 py-1.5 text-xs text-ink shadow-lg tnum"
          style={{ left: hover.x + 12, top: hover.y + 14 }}>
          {hover.text}
        </div>
      )}
    </div>
  );
}

export function GanttLegend() {
  const item = (swatch: React.ReactNode, label: string) => (
    <span className="inline-flex items-center gap-1.5">
      <svg width="18" height="10" aria-hidden>{swatch}</svg>
      {label}
    </span>
  );
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-2">
      {item(<rect width="18" height="10" rx="2" fill="var(--run)" />, "Running")}
      {item(<rect y="3" width="18" height="4" fill="var(--ink-2)" opacity="0.5" />, "Waiting (ready)")}
      {item(
        <>
          <defs>
            <pattern id="lg-io" width="4" height="4" patternUnits="userSpaceOnUse"
              patternTransform="rotate(135)">
              <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-2)" />
            </pattern>
          </defs>
          <rect y="1" width="18" height="8" fill="url(#lg-io)" />
        </>,
        "Blocked on I/O",
      )}
      {item(<rect width="18" height="10" rx="2" fill="var(--switch)" />, "Context switch")}
      {item(
        <>
          <defs>
            <pattern id="lg-idle" width="5" height="5" patternUnits="userSpaceOnUse"
              patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" />
            </pattern>
          </defs>
          <rect width="18" height="10" fill="url(#lg-idle)" />
        </>,
        "CPU idle",
      )}
    </div>
  );
}
