"""On-policy batched training with explicit transition budgets and portable runs."""
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
import time
import torch
import yaml

from . import checkpoint
from .artifacts import append_jsonl, write_json
from .environment import BatchedEnvironment
from .evaluation import evaluate
from .mappo import MAPPO
from .rollout import RolloutBuffer


def collect(agent, env, buffer):
    events = []
    for _ in range(buffer.capacity):
        obs, state = env.observations.clone(), env.states.clone()
        action, latent, log_probs = agent.act(obs)
        value = agent.value(state)
        _, next_state, reward, term, trunc, info = env.step(action)
        next_value = torch.where(term, 0., agent.value(next_state))
        buffer.add(observations=obs, states=state, pre_tanh=latent, old_log_probs=log_probs,
                   rewards=reward, old_values=value, next_values=next_value, terminated=term, truncated=trunc)
        done = term | trunc
        events.append((done.clone(), info, env.initial_qpos.clone()))
        env.reset(done)
    return events


def flush_events(path, events, first_step, num_envs):
    # Host transfer occurs once per rollout rather than once per control step.
    masks = torch.stack([x[0] for x in events]).cpu()
    count = 0
    for t, e in masks.nonzero().tolist():
        row = {key: value[e].cpu().tolist() for key, value in events[t][1].items()}
        if not row["success"]:
            row["completion_time_s"] = None
        row.update(environment_steps=first_step+(t+1)*num_envs, world=e,
                   initial_qpos=events[t][2][e].cpu().tolist())
        append_jsonl(path, row)
        count += 1
    return count


def train(run_dir, env_config, simulation, learning, resume=None):
    root = Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError("choose a new run directory; existing outputs are never overwritten")
    if learning.total_steps % simulation.num_envs:
        raise ValueError("total_steps must be divisible by num_envs (counts team transitions across worlds)")
    if learning.video_interval_steps:
        raise ValueError("training video is disabled in this backend; use evaluate --video-dir after checkpointing")
    if learning.tensorboard:
        raise ValueError("initial backend logs JSONL; set tensorboard: false")
    torch.set_num_threads(learning.torch_threads)
    torch.manual_seed(learning.seed)
    steps = updates = episodes = 0
    best = (-1., -float("inf"), -float("inf"))
    generator = torch.Generator(device=simulation.device).manual_seed(learning.seed)
    from contextlib import ExitStack
    with ExitStack() as stack:
        payload = None
        if resume:
            agent, env_config, _, payload = stack.enter_context(checkpoint.loaded(resume, simulation, optimizers=True))
            old_learning = payload["learning"]
            allowed = {"total_steps", "eval_interval_steps", "eval_episodes", "checkpoint_interval_steps"}
            if any(v != old_learning[k] for k, v in learning.to_dict().items() if k not in allowed):
                raise ValueError("resume preserves optimizer, network, device, seed and rollout settings")
            if simulation.to_dict() != payload["simulation"]:
                raise ValueError("resume preserves simulation backend, world count and contact allocation")
            agent.config = learning
            steps, updates, best = payload["steps"], payload["updates"], tuple(payload["best_score"])
            if learning.total_steps <= steps:
                raise ValueError("resume budget must exceed saved cumulative steps")
        env = BatchedEnvironment(env_config, simulation, learning.seed)
        stack.callback(env.close)
        if payload:
            env.sampler.rng.copy_(payload["spawn_rng"].to(env.device))
            env.reset()
            generator.set_state(payload["minibatch_rng"])
            torch.set_rng_state(payload["torch_rng"])
            if env.device.type == "cuda":
                torch.cuda.set_rng_state(payload["cuda_rng"], env.device)
        else:
            agent = MAPPO(**env.dimensions, config=learning)
        root.mkdir(parents=True, exist_ok=False)
        xml = env.oracle.model_xml
        (root/"arena.xml").write_text(xml)
        env_config = replace(env_config, model_path=str(root/"arena.xml"))
        env.config = env.task.config = env.sampler.cfg = env_config
        (root/"environment.yaml").write_text(yaml.safe_dump(env_config.to_dict()))
        (root/"training.yaml").write_text(yaml.safe_dump(dict(simulation=simulation.to_dict(), learning=learning.to_dict())))
        software = {name: version(name) for name in ("torch", "mujoco")}
        if simulation.backend == "mjwarp":
            software.update({name: version(name) for name in ("mujoco-warp", "warp-lang")})
        write_json(root/"manifest.json", dict(backend=simulation.backend, starting_steps=steps,
                   resume_source=str(resume) if resume else None, resume_behavior="new_episodes",
                   dimensions=env.dimensions, simulation=simulation.to_dict(), learning=learning.to_dict(),
                   environment=env_config.to_dict(), software=software))
        def save(name):
            checkpoint.save(root/"checkpoints"/name, agent, env, simulation, steps, updates, best, generator)
        save("latest.pt")
        next_eval = (steps//learning.eval_interval_steps+1)*learning.eval_interval_steps
        next_checkpoint = (steps//learning.checkpoint_interval_steps+1)*learning.checkpoint_interval_steps
        started, initial_steps = time.perf_counter(), steps
        while steps < learning.total_steps:
            length = min(learning.rollout_steps, (learning.total_steps-steps)//simulation.num_envs)
            buffer = RolloutBuffer(length, simulation.num_envs, env.dimensions, env.device)
            first_step = steps
            events = collect(agent, env, buffer)
            metrics = agent.update(buffer, generator)
            steps += length*simulation.num_envs
            updates += 1
            episodes += flush_events(root/"episodes.jsonl", events, first_step, simulation.num_envs)
            elapsed = time.perf_counter()-started
            append_jsonl(root/"updates.jsonl", dict(environment_steps=steps, update=updates,
                         episodes_this_run=episodes, wall_seconds=elapsed,
                         environment_transitions_per_second=(steps-initial_steps)/elapsed, **metrics))
            print(f"steps={steps}/{learning.total_steps} update={updates} "
                  f"transitions/s={(steps-initial_steps)/elapsed:.1f}", flush=True)
            final = steps == learning.total_steps
            if steps >= next_eval or final:
                # Evaluation is deterministic and uses a separate spawn RNG.
                result = evaluate(agent, env_config, simulation, learning.eval_episodes, seed=1000000)
                result["environment_steps"] = steps
                s = result["summary"]
                score = (s["success_rate"], -s["mean_completion_time_s"] if s["mean_completion_time_s"] is not None
                         else -float("inf"), s["mean_return"])
                improved = score > best
                if improved:
                    best = score
                    save("best.pt")
                    write_json(root/"evaluation_best.json", result)
                write_json(root/"evaluation_latest.json", result)
                append_jsonl(root/"evaluations.jsonl", result)
                print(f"evaluation success={s['success_rate']:.1%} completion={s['mean_completion_time_s']}", flush=True)
                next_eval = (steps//learning.eval_interval_steps+1)*learning.eval_interval_steps
            if steps >= next_checkpoint or final:
                save("latest.pt")
                next_checkpoint = (steps//learning.checkpoint_interval_steps+1)*learning.checkpoint_interval_steps
        return dict(environment_steps=steps, updates=updates, best_success_rate=best[0])
