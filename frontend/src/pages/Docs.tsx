import { Page, Panel } from "../components/ui";
import { useApp } from "../state/AppState";

export function Docs() {
  const { config } = useApp();
  const r = config?.reward_defaults;
  return (
    <Page title="How NeuralOS works" intro="The simulation model, the algorithms, the RL design, the metrics, and what this lab cannot tell you.">
      <Panel>
        <article className="prose-lab">
          <h2 style={{ marginTop: 0 }}>Simulation model</h2>
          <p>
            Time is counted in integer ticks and the engine jumps from event to event: a process arriving, an I/O burst finishing, or a CPU slice ending. Each process has an arrival time, a priority and a burst list alternating CPU and I/O that starts and ends with CPU. A process is NEW until it arrives, then moves between READY, RUNNING and BLOCKED until it is TERMINATED.
          </p>
          <p>When several things happen at the same tick, the order is fixed so every run is reproducible:</p>
          <ul>
            <li>the running slice that ends at that tick stops;</li>
            <li>new arrivals join the ready queue, lowest process id first;</li>
            <li>processes whose I/O finished join next, lowest id first;</li>
            <li>a process whose quantum expired, or that was preempted, joins last.</li>
          </ul>
          <p>
            A context switch is counted whenever the CPU loads a different process from the one that ran last; the first dispatch and re-dispatching the same process are not switches. With a non-zero switch cost the CPU spends that many ticks switching, and the incoming process keeps waiting during the switch. I/O devices are unlimited: blocked processes do their I/O in parallel.
          </p>

          <h2>Baseline schedulers</h2>
          <ul>
            <li><strong>FCFS</strong> runs the oldest ready process to the end of its CPU burst.</li>
            <li><strong>SJF</strong> picks the shortest current CPU burst and runs it to completion (non-preemptive).</li>
            <li><strong>SRTF</strong> is preemptive SJF: a newly ready process with strictly less remaining burst time takes the CPU.</li>
            <li><strong>Round Robin</strong> gives each process at most one quantum, then sends it to the back of the queue.</li>
            <li><strong>Priority</strong> runs the lowest priority number first, non-preemptively. <strong>Priority (preemptive)</strong> lets a strictly higher-priority arrival take the CPU. Optional aging improves a waiting process's priority by one step per configured number of ticks.</li>
            <li><strong>Random</strong> picks a ready process with a seeded random generator. It is a floor any learned policy must clear.</li>
          </ul>
          <p>
            Ties go to whoever joined the ready queue first. SJF, SRTF and the RL agent all see true burst lengths (the textbook assumption); a real OS has to estimate them.
          </p>

          <h2>The RL agent</h2>
          <h3>Observation</h3>
          <p>
            A fixed-size vector describing the {config?.space_defaults.slots ?? 10} oldest ready processes (in queue order) plus global state. Per process: remaining time in the current burst, total remaining CPU work, current and total waiting time, time since arrival, normalised priority, whether it has run yet, CPU bursts left, and whether it was the last process on the CPU. Global: queue length, how many ready processes are hidden beyond the visible slots, the blocked, not-yet-arrived and finished fractions, time to the next event, and the switch cost. Times are divided by the workload's mean CPU burst and log-compressed so every value stays bounded. Showing the oldest processes means a long-waiting process is never invisible to the agent.
          </p>
          <h3>Action</h3>
          <p>
            Choose a visible ready process and a quantum ({(config?.space_defaults.quanta ?? [2, 6, null]).map((q) => (q === null ? "rest of burst" : `${q} ticks`)).join(", ")}). Actions pointing at empty slots are masked, and the policy is trained with MaskablePPO (PPO from sb3-contrib with invalid-action masking), so it never picks one. The environment skips idle gaps, so the agent is only asked when something is ready.
          </p>
          <h3>Reward</h3>
          <p>The environment tracks a running cost and pays its negative increase after each decision:</p>
          <pre><code>{`cost = wait   × total waiting time
     + fair   × Σ (each uninterrupted wait)² / (2 × mean burst)
     + resp   × total response time
     + switch × context switches × mean burst
reward = −Δcost / (processes × mean burst)`}</code></pre>
          <p>
            Defaults: wait {r?.wait ?? 1}, fair {r?.fair ?? 0.1}, resp {r?.resp ?? 0.25}, switch {r?.switch ?? 0.1}. Because the reward telescopes, an episode's return equals the negative total cost, so the agent cannot game it by shifting cost between steps. The squared term is what discourages starvation: one process waiting 100 ticks costs far more than ten processes waiting 10. Turnaround is waiting plus fixed CPU and I/O work, so it is covered by the waiting term. Throughput only moves through idle time and switching, and the switch term covers the part the agent controls.
          </p>
          <p>
            The default fairness weight was calibrated so that FCFS, SJF and Round Robin score within about 15% of each other on validation. No classic policy is already optimal, so the agent has to trade efficiency against fairness. Setting it to 0 gives a plain average-waiting objective; the ablation experiments compare 0, 0.1 and 0.25.
          </p>
          <h3>Training, validation and test data</h3>
          <p>
            Training samples a workload type, size (10–40 processes), switch cost (0 or 1) and seed from 0–4,999,999 every episode, using five workload types. Checkpoints are selected on a fixed validation set (seeds from 5,000,000). Benchmarks use test seeds from 10,000,000, plus two workload types never used in training (long jobs and mixed). Results on those held-out types measure generalisation.
          </p>

          <h2>Metrics</h2>
          <ul>
            <li><strong>Waiting</strong>: time spent ready but not running (turnaround minus CPU and I/O work).</li>
            <li><strong>Turnaround</strong>: completion minus arrival. <strong>Response</strong>: first run minus arrival.</li>
            <li><strong>Throughput</strong>: processes completed per 100 ticks between the first arrival and the last completion. <strong>CPU utilization</strong>: share of that span spent executing processes (not switching).</li>
            <li><strong>Longest wait</strong> and <strong>starved processes</strong>: a process starves if one uninterrupted wait exceeds the threshold, which defaults to 10 × the mean CPU burst.</li>
            <li><strong>Fairness</strong>: Jain's index over slowdowns (turnaround ÷ service time); 1.0 means every process was slowed down equally.</li>
            <li><strong>Objective</strong>: the RL episode cost above, computed identically for every scheduler.</li>
          </ul>

          <h2>Reading the results honestly</h2>
          <ul>
            <li>Every number in the dashboard comes from the simulator. The app shows no illustrative data; panels are empty until something has run.</li>
            <li>The agent optimises a weighted objective, so it should not be expected to beat SRTF on average waiting or Round Robin on longest wait. Look at the trade-off and at the head-to-head counts.</li>
            <li>A single trace is anecdotal. Use the multi-seed benchmarks, which report means with 95% confidence intervals.</li>
            <li>When no model is loaded, the RL agent is disabled everywhere; nothing is ever substituted for it.</li>
          </ul>

          <h2>Limitations</h2>
          <ul>
            <li>One CPU, unlimited I/O devices, no memory, caches or interrupts.</li>
            <li>Burst lengths are known exactly, which flatters SJF, SRTF and the agent compared with a real kernel.</li>
            <li>The agent is trained on synthetic workloads; its behaviour outside them (different sizes, loads or shapes) is not guaranteed.</li>
            <li>The policy has an MLP over a fixed number of slots; very long queues are summarised, not seen in full.</li>
          </ul>

          <h2>Reproducing results</h2>
          <pre><code>{`python -m neuralos.rl.train --run-id ppo-main --timesteps 400000 --seed 0
python -m neuralos.bench.benchmark --config experiments/configs/main.json --model ppo-main
python -m neuralos.rl.evaluate --run-id ppo-main`}</code></pre>
          <p>
            Each saved experiment stores its full configuration and seeds; the Reproduce button re-runs it and checks that every row matches exactly. Training itself is seeded, but PyTorch does not guarantee bit-identical results across machines and thread counts, so a retrained model can score slightly differently.
          </p>
        </article>
      </Panel>
    </Page>
  );
}
