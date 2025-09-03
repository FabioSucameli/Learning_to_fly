import os
import time
import glob
from datetime import datetime
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import gym
import wandb

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
import gym_pybullet_drones as gpd
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
import matplotlib.pyplot as plt

AGGR_PHY_STEPS = 5 # Numero di passi fisici del simulatore tra due step logici dell'agente

#######  utility di I/O
# Crea e restituisce la cartella di output per i video/registrazioni
def ensure_video_dir() -> Path:
    repo_root = Path(gpd.__file__).resolve().parents[1]
    vid_dir = repo_root / "files" / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Video output directory: {vid_dir}")
    return vid_dir

# Estrae i parametri chiave dal nome cartella di un esperimento.
def parse_experiment_path(exp_path: str) -> Dict[str, str]:
    parts = os.path.basename(exp_path).split("-")
    if len(parts) < 6:
        raise ValueError(f"Invalid experiment path format: {exp_path}")
    return {
        "env": parts[1],
        "algo": parts[2],
        "obs": parts[3],
        "act": parts[4],
        "timestamp": "-".join(parts[5:])
    }

# Mappa la stringa CLI al corrispondente ActionType dell'ambiente.
def get_action_type(act_str: str) -> ActionType:
    mapping = {
        'rpm': ActionType.RPM,
        'dyn': ActionType.DYN,
        'pid': ActionType.PID,
        'vel': ActionType.VEL,
        'tun': ActionType.TUN,
        'one_d_rpm': ActionType.ONE_D_RPM,
        'one_d_dyn': ActionType.ONE_D_DYN,
        'one_d_pid': ActionType.ONE_D_PID
    }
    if act_str not in mapping:
        raise ValueError(f"Unknown action type: {act_str}")
    return mapping[act_str]

# Carica un modello SB3 da exp_path
def load_model(exp_path: str, algo: str):
    for name in ['best_model.zip', 'final_model.zip', 'success_model.zip']:
        p = os.path.join(exp_path, name)
        if os.path.isfile(p):
            model_path = p
            break
    else:
        raise FileNotFoundError(f"No model found in {exp_path}. Looked for: ['best_model.zip','final_model.zip','success_model.zip']")
    print(f"[INFO] Loading model from: {model_path}")
    if algo == 'a2c':
        return A2C.load(model_path)
    if algo == 'ppo':
        return PPO.load(model_path)
    if algo == 'sac':
        return SAC.load(model_path)
    raise ValueError(f"Unsupported algorithm: {algo}")

# Reset compatibile con Gym/Gymnasium (gestisce tuple e argomento seed).
# Ritorna solo l'osservazione, scartando info aggiuntive se presenti.
def _safe_reset(env: gym.Env, seed: Optional[int] = None):
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    return out[0] if isinstance(out, tuple) and len(out) >= 1 else out

#Step compatibile con Gym/Gymnasium (gestisce terminated/truncated).
def _safe_step(env: gym.Env, action) -> Tuple[np.ndarray, float, bool, dict]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        return obs, float(reward), done, info
    obs, reward, done, info = out  # type: ignore
    return obs, float(reward), bool(done), info

# Valutazione numerica
# Valuta un modello su n_episodes e restituisce statistiche dettagliate.
def evaluate_model(model, env, n_episodes: int = 10, *, deterministic: bool = True,
                   seed_base: Optional[int] = None) -> Dict[str, Any]:
    print(f"[INFO] Evaluating model over {n_episodes} episodes (deterministic={deterministic})...")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_episodes, deterministic=deterministic
    )

    # Raccolta dettagli episodio per episodio
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    for ep in range(n_episodes):
        seed = (seed_base + ep) if seed_base is not None else None
        obs = _safe_reset(env, seed)
        ep_r, ep_len, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, info = _safe_step(env, action)
            ep_r += float(r)
            ep_len += 1
        episode_rewards.append(float(ep_r))
        episode_lengths.append(int(ep_len))

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "min_reward": float(np.min(episode_rewards)) if episode_rewards else float("nan"),
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else float("nan"),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else float("nan"),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }



# Esegue un breve rollout con render/recording e genera un grafico di take-off/hover.
# Calcola quota z(t) e velocità , decollo e stabilizzazione
def run_visual_test(model, env, obs_type: ObservationType, duration: int = 6, *, deterministic: bool = True) \
        -> Tuple[Optional[str], float]:
    print(f"[INFO] Running visual test for {duration} seconds (deterministic={deterministic})...")

    obs = _safe_reset(env)
    start = time.time()

    steps = duration * int(env.SIM_FREQ / env.AGGR_PHY_STEPS)
    total_r = 0.0

    ts_list, z_list, vz_list = [], [], []
    dt = float(env.AGGR_PHY_STEPS) / float(env.SIM_FREQ)

    # per la derivata numerica se serve
    last_z = None
    last_t = None

    for i in range(steps):
        # azione e step
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, done, info = _safe_step(env, action)
        total_r += float(r)

        # stato "vero" da PyBullet
        state = env.unwrapped._getDroneStateVector(0)

        # mapping
        z_true = float(state[2])          # quota
        vz_raw = float(state[12])         # velocità verticale

        # fallback: se per qualche motivo è zero/NaN, calcolo la derivata
        t_now = i * dt
        if not np.isfinite(vz_raw) or abs(vz_raw) < 1e-10:
            if last_z is not None:
                vz_raw = (z_true - last_z) / (t_now - last_t)
            else:
                vz_raw = 0.0
        last_z, last_t = z_true, t_now

        ts_list.append(t_now)
        z_list.append(z_true)
        vz_list.append(vz_raw)

        # render non-bloccante
        try:
            env.render()
        except Exception:
            pass        

        sync(np.floor(i * env.AGGR_PHY_STEPS), start, env.TIMESTEP)

    print(f"[INFO] Visual test completed. Total reward: {total_r:.4f}")

    # — grafico take-off + hover —
    try:
        ts = np.asarray(ts_list)
        z  = np.asarray(z_list)
        vz = np.asarray(vz_list)

        # target z: da env se disponibile; altrimenti dalla mediana finale
        target_z = None
        for attr in ("TARGET_POS", "TARGET_POS_HOVER", "HOVERING_Z", "TARGET_HEIGHT"):
            if hasattr(env, attr):
                val = getattr(env, attr)
                try:
                    target_z = float(val[2])
                except Exception:
                    try:
                        target_z = float(val)
                    except Exception:
                        pass
                if target_z is not None:
                    break
        if target_z is None:
            tail = max(10, int(0.3 * len(z)))
            target_z = float(np.median(z[-tail:]))

        # rilevamento inizio hover
        band_z, band_v = 0.015, 0.02
        win = max(8, int(0.5 / dt))
        ok = (np.abs(z - target_z) <= band_z) & (np.abs(vz) <= band_v)
        idx_hover = next((k for k in range(len(z) - win) if np.all(ok[k:k + win])), len(z) - 1)

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.plot(ts, z, lw=2.4, color="#1f77b4", label="z(t) [m]")
        ax1.axhline(target_z, ls=(0, (6, 6)), lw=1.6, color="#7f8c8d", alpha=0.9, label="target altitude")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Altitude z [m]")

        ax1.axvspan(ts[0], ts[idx_hover], facecolor="#fde68a", alpha=0.35, label="take-off")
        if idx_hover != len(z) - 1:
            ax1.axvline(ts[idx_hover], color="#d35400", lw=1.2, ls=":", label="hover start")
        else:
            print("[INFO] Hover not detected within the time window: consider increasing duration or relaxing thresholds.")

        ax2 = ax1.twinx()
        ax2.plot(ts, vz, lw=1.8, ls="--", color="#16a085", label="ż(t) [m/s]")
        ax2.set_ylabel("Vertical speed ż [m/s]")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        ax1.set_title("Take-off (reaching altitude) + Hover (stabilization)")
        plt.tight_layout()

        out_png = f"takeoff_hover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out_png, dpi=220)
        plt.close(fig)
        print(f"[INFO] Saved plot: {out_png} | hover_start={ts[idx_hover]:.2f}s, z_ss≈{target_z:.3f} m")
        print(f"[DBG] z_final={z[-1]:.3f} m, target_z={target_z:.3f} m")
    except Exception as e:
        print(f"[WARNING] plot fail: {e}")

    return float(total_r)

#### wandb
#Inizializza una run di test e restituisce il nome della run
def setup_wandb_test(exp_params: Dict[str, str], project: str) -> str:
    run_name = f"test-{exp_params['env']}-{exp_params['algo']}-{exp_params['timestamp']}"
    wandb.init(
        project=project,
        name=run_name,
        config={
            "mode": "testing",
            "algorithm": exp_params["algo"],
            "environment": exp_params["env"],
            "observation_type": exp_params["obs"],
            "action_type": exp_params["act"],
            "timestamp": exp_params["timestamp"],
        },
        tags=["testing", exp_params["env"], exp_params["algo"]],
    )
    return run_name

# Crea e registra su W&B tabelle e grafici riassuntivi dei risultati di valutazione.
# Summary: metriche aggregate (mean/best/worst reward, lunghezze episodio).
# Grafici: bar chart comparativo e istogramma della distribuzione dei reward.
def log_eval_plots_to_wandb(eval_res: Dict[str, Any], *, title_prefix: str = "eval"):

    ep_rewards = eval_res["episode_rewards"]
    ep_lengths = eval_res["episode_lengths"]
    n = len(ep_rewards)

    # Summary "card" (utile per ordinare/scremare run nella UI)
    wandb.summary[f"{title_prefix}/mean_episode_reward"] = eval_res["mean_reward"]
    wandb.summary[f"{title_prefix}/best_episode_reward"] = eval_res["max_reward"]
    wandb.summary[f"{title_prefix}/worst_episode_reward"] = eval_res["min_reward"]
    wandb.summary[f"{title_prefix}/mean_episode_length"] = eval_res["mean_episode_length"]
    wandb.summary[f"{title_prefix}/std_episode_length"] = eval_res["std_episode_length"]

    # Tabella per episodio
    table = wandb.Table(columns=["episode", "reward", "length"])
    for i, (r, L) in enumerate(zip(ep_rewards, ep_lengths)):
        table.add_data(i, float(r), int(L))
    wandb.log({f"{title_prefix}/episodes_table": table})

    # Bar chart (mean/best/worst)
    bar_data = wandb.Table(
        data=[
            ["mean", float(eval_res["mean_reward"])],
            ["best", float(eval_res["max_reward"])],
            ["worst", float(eval_res["min_reward"])],
        ],
        columns=["metric", "value"],
    )
    wandb.log({
        f"{title_prefix}/reward_summary_bar":
            wandb.plot.bar(bar_data, "metric", "value", title="Episode Reward Summary")
    })

    # Istogramma della distribuzione dei reward
    hist_table = wandb.Table(data=[[float(x)] for x in ep_rewards], columns=["reward"])
    wandb.log({
        f"{title_prefix}/reward_hist":
            wandb.plot.histogram(hist_table, "reward", title="Episode Reward Distribution")
    })

    # Line chart reward per episodio
    xs = list(range(n))
    wandb.log({
        f"{title_prefix}/episode_reward_line":
            wandb.plot.line_series(xs=xs, ys=[ep_rewards], keys=["reward"],
                                   title="Episode Reward by Episode", xname="episode")
    })



# Esegue la pipeline completa di test per una singola cartella esperimento.
def test_single_experiment(exp_path: str, n_episodes: int = 10,
                           headless: bool = False, visual_duration: int = 6,
                           project: str = "drone-rl-testing",
                           *, deterministic: bool = True, seed_base: Optional[int] = None)\
        -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Testing experiment: {os.path.basename(exp_path)}")
    print(f"{'='*60}")

    #Parametri esperimento + run wandb
    exp_params = parse_experiment_path(exp_path)
    run_name = setup_wandb_test(exp_params, project)

    # Modello + ambiente di valutazione
    model = load_model(exp_path, exp_params['algo'])

    env_name = f"{exp_params['env']}-aviary-v0"
    obs_type = ObservationType.KIN if exp_params['obs'] == 'kin' else ObservationType.RGB
    act_type = get_action_type(exp_params['act'])

    eval_env = gym.make(
        env_name,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=obs_type,
        act=act_type,
        gui=False,
        record=False
    )
    eval_env = Monitor(eval_env)

    # Valutazione + logging
    eval_results = evaluate_model(model, eval_env, n_episodes,
                                  deterministic=deterministic, seed_base=seed_base)

    # Log delle metriche e grafici "nuovi"
    log_eval_plots_to_wandb(eval_results, title_prefix="eval")

    print(f"\n[EVALUATION RESULTS]")
    print(f"Mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"Best/Worst: {eval_results['max_reward']:.4f} / {eval_results['min_reward']:.4f}")
    print(f"Mean episode length: {eval_results['mean_episode_length']:.1f} ± {eval_results['std_episode_length']:.1f}")

    # Visual test (opzionale) con gui
    visual_reward: Optional[float] = None
    log_path: Optional[str] = None

    if not headless:
        print(f"\n[VISUAL TEST]")
        ensure_video_dir()
        visual_env = gym.make(
            env_name,
            gui=True,
            record=True,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            obs=obs_type,
            act=act_type
        )
        try:
            log_path, visual_reward = run_visual_test(
                model, visual_env, obs_type, visual_duration, deterministic=deterministic
            )
            wandb.log({
                "visual_test/reward": visual_reward,
                "visual_test/duration_seconds": visual_duration,
                "visual_test/log_csv": log_path if log_path is not None else "N/A",
            })
        except Exception as e:
            print(f"[WARNING] Visual test failed: {e}")
        finally:
            try:
                visual_env.close()
            except Exception:
                pass
    
    # Riepilogo e chiusura risorse.
    try:
        eval_env.close()
    except Exception:
        pass

    results: Dict[str, Any] = {
        "experiment_path": exp_path,
        "algorithm": exp_params['algo'],
        "environment": exp_params['env'],
        "run_name": run_name,
        **eval_results
    }
    if visual_reward is not None:
        results["visual_test_reward"] = float(visual_reward)
    if log_path is not None:
        results["log_path"] = str(log_path)

    wandb.finish()
    return results

#### Ricerca/confronto batch
def find_experiments(exp_dir: str, pattern: str) -> List[str]:
    found = glob.glob(os.path.join(exp_dir, pattern))
    dirs = [p for p in found if os.path.isdir(p)]
    dirs.sort()
    return dirs

#Valuta in serie più esperimenti e registra un confronto su W&B.
#Ordina i risultati per reward medio decrescente e pubblica:
#tabella riassuntiva (Rank, Algo, Env, Mean/Std/Min/Max),
#bar chart del mean reward per algoritmo.
def compare_experiments(experiment_paths: List[str], n_episodes: int = 10,
                        project: str = "drone-rl-testing",
                        *, deterministic: bool = True, seed_base: Optional[int] = None)\
        -> List[Dict[str, Any]]:
    print(f"\n{'='*80}")
    print(f"COMPARING {len(experiment_paths)} EXPERIMENTS")
    print(f"{'='*80}")

    results: List[Dict[str, Any]] = []
    for exp_path in experiment_paths:
        try:
            res = test_single_experiment(
                exp_path, n_episodes, headless=True, project=project,
                deterministic=deterministic, seed_base=seed_base
            )
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed to test {exp_path}: {e}")

    if len(results) > 1:
        results.sort(key=lambda x: x["mean_reward"], reverse=True)

        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"{'Rank':<4} {'Algorithm':<8} {'Environment':<10} {'Mean Reward':<12} {'Std Reward':<12}")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"{i:<4} {r['algorithm'].upper():<8} {r['environment']:<10} {r['mean_reward']:<12.4f} {r['std_reward']:<12.4f}")

        # Run "comparison" con bar chart interattivo
        wandb.init(
            project=project,
            name=f"comparison-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}",
            config={"mode": "comparison", "n_experiments": len(results)},
            tags=["comparison"]
        )

        comp_table = wandb.Table(columns=["Rank", "Algorithm", "Environment",
                                          "Mean Reward", "Std Reward", "Min Reward", "Max Reward"])
        for i, r in enumerate(results, 1):
            comp_table.add_data(i, r["algorithm"], r["environment"],
                                float(r["mean_reward"]), float(r["std_reward"]),
                                float(r["min_reward"]), float(r["max_reward"]))
        wandb.log({"comparison/results_table": comp_table})

        # Bar chart interattivo (mean reward per algoritmo)
        bar_table = wandb.Table(
            data=[[r["algorithm"], float(r["mean_reward"])] for r in results],
            columns=["algorithm", "mean_reward"]
        )
        wandb.log({
            "comparison/mean_reward_bar":
                wandb.plot.bar(bar_table, "algorithm", "mean_reward", title="Mean Episode Reward per Algorithm")
        })

        wandb.finish()

    return results


# CLI
# Entry-point CLI: parsing argomenti, esecuzione test singolo o confronto batch.
def main():
    parser = argparse.ArgumentParser(description="Test trained models with improved W&B logging")
    parser.add_argument('--exp', type=str, default=None, help='Path to single experiment directory to test')
    parser.add_argument('--exp_dir', type=str, default=None, help='Directory containing multiple experiments')
    parser.add_argument('--pattern', type=str, default='save-*', help='Pattern to match experiment directories')

    parser.add_argument('--n_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--headless', action='store_true', help='Run without GUI (no visual test)')
    parser.add_argument('--gui', action='store_true', help='Force GUI on (overrides --headless)')
    parser.add_argument('--visual_duration', type=int, default=6, help='Duration of visual test in seconds')

    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help='Use deterministic policy at evaluation (default)')
    parser.add_argument('--stochastic', dest='deterministic', action='store_false',
                        help='Use stochastic policy at evaluation')
    parser.set_defaults(deterministic=True)
    parser.add_argument('--seed_base', type=int, default=None,
                        help='If set, reset each eval episode with seed_base + i')

    parser.add_argument('--project', type=str, default='drone-rl-testing', help='W&B project name')

    args = parser.parse_args()

    # Coerenza flag GUI/headless
    if args.gui:
        args.headless = False
    # Validazioni di input
    if not args.exp and not args.exp_dir:
        parser.error("Must specify either --exp for single experiment or --exp_dir for multiple experiments")
    # Esecuzione modalità singola o batch
    if args.exp and args.exp_dir:
        parser.error("Cannot specify both --exp and --exp_dir")

    if args.exp:
        if not os.path.exists(args.exp):
            parser.error(f"Experiment directory not found: {args.exp}")
        try:
            test_single_experiment(
                args.exp, args.n_episodes, args.headless, args.visual_duration, args.project,
                deterministic=args.deterministic, seed_base=args.seed_base,
            )
            print("\n[SUCCESS] Testing completed!")
            print(f"Check wandb project '{args.project}' for detailed results.")
        except Exception as e:
            print(f"[ERROR] Testing failed: {e}")
            return 1
    else:
        if not os.path.exists(args.exp_dir):
            parser.error(f"Experiments directory not found: {args.exp_dir}")
        experiments = find_experiments(args.exp_dir, args.pattern)
        if not experiments:
            print(f"[WARNING] No experiments found in {args.exp_dir} matching pattern '{args.pattern}'")
            return 1
        print(f"[INFO] Found {len(experiments)} experiments to test:")
        for p in experiments:
            print(f"  - {os.path.basename(p)}")
        try:
            results = compare_experiments(
                experiments, args.n_episodes, args.project,
                deterministic=args.deterministic, seed_base=args.seed_base
            )
            print("\n[SUCCESS] Comparison completed!")
            print(f"Tested {len(results)} experiments successfully.")
            print(f"Check wandb project '{args.project}' for detailed results and comparisons.")
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            return 1
    return 0


if __name__ == "__main__":
    exit(main())
