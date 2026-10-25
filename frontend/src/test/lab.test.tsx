import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { AppStateProvider } from "../state/AppState";
import srtf from "./fixtures/srtf-classic.json";

type Handler = (init?: RequestInit) => { status?: number; body: unknown };

const CONFIG = {
  schedulers: [], workload_kinds: [], limits: { max_processes: 500 },
  reward_defaults: { wait: 1, fair: 0.1, resp: 0.25, switch: 0.1 },
  space_defaults: { slots: 10, quanta: [2, 6, null] },
  seed_ranges: { train: [0, 4999999], validation: 5000000, test: 10000000 },
};

function mockApi(routes: Record<string, Handler>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const key = Object.keys(routes).find((k) => url.endsWith(k) || url.includes(k + "?"));
    if (!key) return new Response(JSON.stringify({ detail: `no mock for ${url}` }), { status: 404 });
    const { status = 200, body } = routes[key]!(init);
    return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

const BASE: Record<string, Handler> = {
  "/api/health": () => ({ body: { status: "ok", version: "1.0.0", rl_available: true, missing_dependencies: [] } }),
  "/api/config": () => ({ body: CONFIG }),
  "/api/models/status": () => ({ body: {
    state: "unavailable", active: null, training: [], missing_dependencies: [],
    available_checkpoints: [], message: "No trained model is loaded.",
  } }),
  "/api/experiments": () => ({ body: { experiments: [] } }),
};

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppStateProvider>
        <App />
      </AppStateProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.useRealTimers();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("simulation lab", () => {
  it("runs a simulation against the API and steps through the real trace", async () => {
    const calls = mockApi({ ...BASE, "/api/simulations": () => ({ body: srtf }) });
    const user = userEvent.setup();
    renderAt("/lab");

    // RL agent cannot be chosen while no model is loaded.
    await waitFor(() => expect(screen.getByRole("option", { name: /RL agent \(load a model first\)/ })).toBeDisabled());

    await user.click(screen.getByRole("tab", { name: "Hand-built processes" }));
    await user.selectOptions(screen.getByLabelText("Scheduler"), "srtf");
    await user.click(screen.getByRole("button", { name: "Run simulation" }));

    const post = calls.find((c) => c.url.endsWith("/api/simulations"));
    expect(post?.init?.method).toBe("POST");
    const body = JSON.parse(String(post?.init?.body));
    expect(body.scheduler).toEqual({ name: "srtf" });
    expect(body.workload.processes).toHaveLength(4);

    // Average waiting 6.5 from the real SRTF result.
    expect(await screen.findByText("6.5")).toBeInTheDocument();

    const cpu = () => screen.getByTestId("cpu-now").textContent;
    expect(cpu()).toBe("Finished");
    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(cpu()).toBe("Running P1");
    await user.click(screen.getByRole("button", { name: "Step to next event" }));
    expect(cpu()).toBe("Running P2"); // P2 preempts P1 at t=1
    await user.click(screen.getByRole("button", { name: "Step to next event" }));
    await user.click(screen.getByRole("button", { name: "Step to next event" }));
    await user.click(screen.getByRole("button", { name: "Step to next event" }));
    expect(cpu()).toBe("Running P4"); // t=5
    await user.click(screen.getByRole("button", { name: "Step back to previous event" }));
    expect(cpu()).toBe("Running P2");

    await user.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Pause" }));

    // Process inspector shows the real state history.
    await user.click(screen.getAllByRole("button", { name: "P1" }).at(-1)!);
    const detail = await screen.findByRole("heading", { name: "Process P1" });
    const panel = detail.closest("section")!;
    expect(within(panel).getByText("t=17")).toBeInTheDocument(); // terminated at 17
  });

  it("shows the API's reason when a request is refused", async () => {
    mockApi({
      ...BASE,
      "/api/simulations": () => ({ status: 409, body: { detail: "No trained model is loaded. Train one first." } }),
    });
    const user = userEvent.setup();
    renderAt("/lab");
    await user.click(await screen.findByRole("button", { name: "Run simulation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No trained model is loaded. Train one first.");
  });

  it("reports an unreachable API", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
    renderAt("/");
    expect(await screen.findByText(/The API is not reachable/)).toBeInTheDocument();
  });
});

describe("model status", () => {
  it("distinguishes loaded, training and unavailable models", async () => {
    mockApi({
      ...BASE,
      "/api/models/status": () => ({ body: {
        state: "loaded", training: [], missing_dependencies: [], available_checkpoints: ["ppo-main"],
        message: "A trained checkpoint is loaded.",
        active: { run_id: "ppo-main", checkpoint: "best", loaded_at: "", algorithm: "MaskablePPO",
          seed: 0, timesteps: 400000, status: "completed", best: null },
      } }),
    });
    renderAt("/");
    await waitFor(() => expect(screen.getAllByTestId("model-badge")[0]).toHaveTextContent("Trained model loaded"));
    expect((await screen.findAllByText(/ppo-main/)).length).toBeGreaterThan(0);
  });

  it("offers a sample run when nothing has been simulated", async () => {
    mockApi(BASE);
    renderAt("/");
    expect(await screen.findByRole("button", { name: "Run a sample simulation" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByTestId("model-badge")[0]).toHaveTextContent("No model loaded"));
  });
});
