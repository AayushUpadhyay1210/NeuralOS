// Pure helpers that replay a finished simulation's real transition log.
// The UI never invents states: everything here is derived from backend output.

import type { ProcessStateName, Segment, TransitionOut } from "../api/types";

export interface StateSpan {
  pid: number;
  start: number;
  end: number;
  state: ProcessStateName;
}

/** Per-process state intervals from the transition log (NEW before arrival, ends at completion). */
export function stateSpans(transitions: TransitionOut[]): Map<number, StateSpan[]> {
  const byPid = new Map<number, TransitionOut[]>();
  for (const t of transitions) {
    const list = byPid.get(t.pid) ?? [];
    list.push(t);
    byPid.set(t.pid, list);
  }
  const out = new Map<number, StateSpan[]>();
  for (const [pid, list] of byPid) {
    const spans: StateSpan[] = [];
    for (let i = 0; i < list.length; i++) {
      const cur = list[i]!;
      if (cur.to === "TERMINATED") break;
      const next = list[i + 1];
      const end = next ? next.time : cur.time;
      if (end > cur.time) spans.push({ pid, start: cur.time, end, state: cur.to });
    }
    out.set(pid, spans);
  }
  return out;
}

export interface Snapshot {
  time: number;
  states: Map<number, ProcessStateName>;
  /** READY processes in the order they joined the queue. */
  readyQueue: number[];
  running: number | null;
  cpu: Segment | null;
}

/**
 * State of every process at time ``t``. Transitions at exactly ``t`` are
 * applied, so the snapshot shows the situation just after the instant.
 */
export function snapshotAt(
  t: number,
  transitions: TransitionOut[],
  timeline: Segment[],
  pids: number[],
): Snapshot {
  const states = new Map<number, ProcessStateName>(pids.map((p) => [p, "NEW"]));
  const readyOrder = new Map<number, number>();
  transitions.forEach((tr, idx) => {
    if (tr.time > t) return;
    states.set(tr.pid, tr.to);
    if (tr.to === "READY") readyOrder.set(tr.pid, idx);
    else readyOrder.delete(tr.pid);
  });
  const readyQueue = [...readyOrder.entries()].sort((a, b) => a[1] - b[1]).map(([pid]) => pid);
  let running: number | null = null;
  for (const [pid, s] of states) if (s === "RUNNING") running = pid;
  const cpu = timeline.find((s) => s.start <= t && t < s.end) ?? null;
  return { time: t, states, readyQueue, running, cpu };
}

/** Sorted distinct instants at which something happens (for single-stepping). */
export function eventTimes(transitions: TransitionOut[], timeline: Segment[]): number[] {
  const set = new Set<number>([0]);
  for (const t of transitions) set.add(t.time);
  for (const s of timeline) {
    set.add(s.start);
    set.add(s.end);
  }
  return [...set].sort((a, b) => a - b);
}

export function nextEvent(times: number[], t: number): number {
  for (const x of times) if (x > t) return x;
  return times[times.length - 1] ?? t;
}

export function prevEvent(times: number[], t: number): number {
  let best = 0;
  for (const x of times) if (x < t) best = x;
  return best;
}

export interface IntervalSummary {
  start: number;
  end: number;
  busy: number;
  idle: number;
  switching: number;
  byPid: { pid: number; ticks: number }[];
}

/** CPU usage inside [start, end) computed from the real timeline. */
export function summariseInterval(timeline: Segment[], start: number, end: number): IntervalSummary {
  let busy = 0;
  let idle = 0;
  let switching = 0;
  const per = new Map<number, number>();
  for (const s of timeline) {
    const a = Math.max(s.start, start);
    const b = Math.min(s.end, end);
    if (b <= a) continue;
    const d = b - a;
    if (s.kind === "run") {
      busy += d;
      if (s.pid !== null) per.set(s.pid, (per.get(s.pid) ?? 0) + d);
    } else if (s.kind === "idle") idle += d;
    else switching += d;
  }
  const byPid = [...per.entries()]
    .map(([pid, ticks]) => ({ pid, ticks }))
    .sort((a, b) => b.ticks - a.ticks || a.pid - b.pid);
  return { start, end, busy, idle, switching, byPid };
}
