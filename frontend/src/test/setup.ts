import "@testing-library/jest-dom/vitest";

// jsdom lacks ResizeObserver, used by the Gantt trace and Recharts.
class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver = RO;
