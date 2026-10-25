import { describe, expect, it } from "vitest";
import type { SimulationResult } from "../api/types";
import { eventTimes, nextEvent, prevEvent, snapshotAt, stateSpans, summariseInterval } from "../lib/trace";
import fcfsIo from "./fixtures/fcfs-io.json";

// Real backend output: FCFS, switch cost 1.
// P1 runs 0-2, blocks 2-5; switch 2-3; P2 3-7; switch 7-8; P1 8-10; switch 10-11; P3 11-12.
const sim = fcfsIo as unknown as SimulationResult;
const pids = [1, 2, 3];

describe("trace replay", () => {
  it("derives per-process state spans from transitions", () => {
    const spans = stateSpans(sim.transitions).get(1)!;
    expect(spans.map((s) => [s.state, s.start, s.end])).toEqual([
      ["RUNNING", 0, 2],
      ["BLOCKED", 2, 5],
      ["READY", 5, 8],
      ["RUNNING", 8, 10],
    ]);
  });

  it("snapshots the CPU and queues at an instant", () => {
    const s = snapshotAt(6, sim.transitions, sim.timeline, pids);
    expect(s.running).toBe(2);
    expect(s.readyQueue).toEqual([1]);
    expect(s.states.get(3)).toBe("NEW");
    expect(s.cpu).toMatchObject({ kind: "run", pid: 2 });
  });

  it("shows a context switch with the incoming process still ready", () => {
    const s = snapshotAt(2.5, sim.transitions, sim.timeline, pids);
    expect(s.cpu).toMatchObject({ kind: "switch", pid: 2 });
    expect(s.states.get(1)).toBe("BLOCKED");
    expect(s.readyQueue).toEqual([2]);
    expect(s.running).toBeNull();
  });

  it("marks everything finished after the end", () => {
    const s = snapshotAt(12, sim.transitions, sim.timeline, pids);
    expect([...s.states.values()]).toEqual(["TERMINATED", "TERMINATED", "TERMINATED"]);
    expect(s.cpu).toBeNull();
  });

  it("summarises an interval", () => {
    const all = summariseInterval(sim.timeline, 0, 12);
    expect(all).toMatchObject({ busy: 9, switching: 3, idle: 0 });
    expect(all.byPid).toEqual([{ pid: 1, ticks: 4 }, { pid: 2, ticks: 4 }, { pid: 3, ticks: 1 }]);
    const part = summariseInterval(sim.timeline, 6, 9);
    expect(part).toMatchObject({ busy: 2, switching: 1 });
  });

  it("steps between event times", () => {
    const times = eventTimes(sim.transitions, sim.timeline);
    expect(times).toEqual([0, 2, 3, 5, 7, 8, 9, 10, 11, 12]);
    expect(nextEvent(times, 3)).toBe(5);
    expect(prevEvent(times, 3)).toBe(2);
    expect(nextEvent(times, 12)).toBe(12);
  });
});
