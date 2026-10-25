import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SimulationResult } from "../api/types";
import { eventTimes, nextEvent, prevEvent } from "./trace";

/**
 * Plays back a finished simulation over its real event log.
 * Playback only moves a cursor through results the backend computed.
 */
export function usePlayback(sim: SimulationResult | null) {
  const end = sim?.timeline.length ? sim.timeline[sim.timeline.length - 1]!.end : 0;
  const times = useMemo(
    () => (sim ? eventTimes(sim.transitions, sim.timeline) : [0]),
    [sim],
  );
  const [t, setT] = useState(end);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(10); // ticks per second
  const raf = useRef<number | null>(null);
  const last = useRef<number | null>(null);

  // New result: start at the end so the full picture shows first.
  const [prevSim, setPrevSim] = useState(sim);
  if (sim !== prevSim) {
    setPrevSim(sim);
    setT(end);
    setPlaying(false);
  }

  useEffect(() => {
    if (!playing) return;
    const tick = (now: number) => {
      const prev = last.current ?? now;
      last.current = now;
      setT((cur) => {
        const nt = Math.min(end, cur + ((now - prev) / 1000) * speed);
        if (nt >= end) setPlaying(false);
        return nt;
      });
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
      last.current = null;
    };
  }, [playing, speed, end]);

  const play = useCallback(() => {
    setT((cur) => (cur >= end ? 0 : cur));
    setPlaying(true);
  }, [end]);
  const pause = useCallback(() => setPlaying(false), []);
  const reset = useCallback(() => {
    setPlaying(false);
    setT(0);
  }, []);
  const replay = useCallback(() => {
    setT(0);
    setPlaying(true);
  }, []);
  const step = useCallback(
    (dir: 1 | -1 = 1) => {
      setPlaying(false);
      setT((cur) => (dir === 1 ? nextEvent(times, Math.floor(cur)) : prevEvent(times, Math.ceil(cur))));
    },
    [times],
  );
  const seek = useCallback((v: number) => {
    setPlaying(false);
    setT(Math.max(0, Math.min(end, v)));
  }, [end]);

  return { t, tInt: Math.floor(t), end, playing, speed, setSpeed, play, pause, reset, replay, step, seek };
}
