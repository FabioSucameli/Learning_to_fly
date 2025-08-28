import os
import time
from datetime import datetime
import argparse
import subprocess
import numpy as np
import gym
import torch
import wandb
from typing import Dict, Any, Optional
try:
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    from stable_baselines3.common.cmd_util import make_vec_env

from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy  # <- per eval finale

from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

# Costanti di configurazione (unità: timesteps)
EPISODE_REWARD_THRESHOLD = -0.0  # Soglia di reward per interrompere l’addestramento
TOTAL_TIMESTEPS = 35000          # Timesteps massimi per algoritmo (per confronti uniformi)
EVAL_FREQ_DIVISOR = 2000         # Una valutazione ogni (EVAL_FREQ_DIVISOR / cpu) step reali
AGGR_PHY_STEPS = 5               # Passi fisici aggregati tra due step logici dell’agent

# Callback per registrare su Weights & Biases le metriche di training.
# Pubblica reward e lunghezza episodio solo a fine episodio (chiave 'episode' del Monitor di SB3).
# Mantiene l'asse X clippato a TOTAL_TIMESTEPS per confronti coerenti tra algoritmi/parallellismo.
class WandbCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    # X clippato per confronti coerenti tra algoritmi
                    x_step = min(self.num_timesteps, self.total_timesteps)
                    wandb.log({
                        'train/episode_reward': episode_reward,
                        'train/episode_length': episode_length,
                        'train/timesteps': self.num_timesteps,  # non clippato, utile per diagnostica
                        'train/x_timesteps': x_step             # clippato: usato come asse X
                    }, step=x_step)
        return True

    def _on_training_end(self) -> None:
        wandb.log({'train/x_timesteps': self.total_timesteps, 'train/done': 1}, step=self.total_timesteps)


# Estende EvalCallback aggiungendo logging su Weights & Biases.
# Sincronizza l'asse X ('eval/x_timesteps') con un contatore clippato a TOTAL_TIMESTEPS.
# Esegue valutazioni periodiche e salva il best model in base alla ricompensa media.
class WandbEvalCallback(EvalCallback):
    
    def __init__(self, *args, total_timesteps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward = np.mean(self.last_mean_reward) if isinstance(self.last_mean_reward, list) else self.last_mean_reward
            std_reward = np.std(self.last_mean_reward) if isinstance(self.last_mean_reward, list) else 0

            x_step = min(self.num_timesteps, self.total_timesteps)
            wandb.log({
                'eval/mean_reward': mean_reward,
                'eval/std_reward': std_reward,
                'eval/timesteps': self.num_timesteps,  # non clippato, per reference
                'eval/x_timesteps': x_step,            # clippato: asse X
                'eval/best_mean_reward': self.best_mean_reward
            }, step=x_step)
        return result

    def _on_training_end(self) -> None:
        super()._on_training_end()

        # Esegue un'ultima valutazione al termine dell'addestramento
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
            render=False,
            return_episode_rewards=False
        )

        # Log finale allineato
        wandb.log({
            'eval/mean_reward': mean_reward,
            'eval/std_reward': std_reward,
            'eval/best_mean_reward': self.best_mean_reward,
            'eval/timesteps': self.total_timesteps,   # non clippato
            'eval/x_timesteps': self.total_timesteps  # clippato/usato come asse X
        }, step=self.total_timesteps)


# Restituisce l'hash del commit Git corrente
def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            ).decode().strip()
        except Exception:
            return "unknown"

# Scelta della classe ambiente in base al nome CLI
# Usa seed esplicito solo se non negativo; altrimenti lascia la randomizzazione di default
def create_environment(
    env_name: str,
    env_kwargs: Dict[str, Any],
    n_envs: int = 1,
    seed: Optional[int] = None,
):
    if env_name == "takeoff-aviary-v0":
        env_cls = TakeoffAviary
    elif env_name == "hover-aviary-v0":
        env_cls = HoverAviary
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    base_seed = None if (seed is None or seed < 0) else seed
    return make_vec_env(env_cls, env_kwargs=env_kwargs, n_envs=n_envs, seed=base_seed)

# Crea e restituisce il modello RL in base ad algoritmo e tipo di osservazione.
# On-policy: architetture simmetriche con testa VF/PI separate (stabile su 'kin' e 'rgb')
# Off-policy (SAC): MLP/CNN più profonda per catturare dinamiche continue
# 'n_steps' opzionale
def create_model(algo: str, policy_type: str, env, config: Dict[str, Any]):
    if algo in ['a2c', 'ppo']:
        # Algoritmo on-policy
        onpolicy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
        )
        if algo == 'a2c':
            model_class = A2C
        else:  # ppo
            model_class = PPO
        policy_class = a2cppoMlpPolicy if policy_type == 'kin' else a2cppoCnnPolicy

        # Opzionale: n_steps
        extra_kwargs = {}
        n_steps = config.get('n_steps', None)
        if n_steps is not None and n_steps > 0:
            extra_kwargs['n_steps'] = n_steps

        return model_class(policy_class, env, policy_kwargs=onpolicy_kwargs, verbose=1, **extra_kwargs)

    elif algo == 'sac':
        # Algoritmo off-policy
        offpolicy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[512, 512, 256, 128]
        )
        policy_class = sacMlpPolicy if policy_type == 'kin' else sacCnnPolicy
        return SAC(policy_class, env, policy_kwargs=offpolicy_kwargs, verbose=1)

    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

#Inizializza Weights & Biases, definisce le metriche/assi X e restituisce il run name.
def setup_wandb(args, algo: str) -> str:
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    run_name = f"{args.env}-{algo}-{args.obs.value}-{args.act.value}-{timestamp}"

    wandb.init(
        project=args.project,
        name=run_name,
        config={
            "algorithm": algo,
            "environment": args.env,
            "observation_type": args.obs.value,
            "action_type": args.act.value,
            "cpu_count": args.cpu,
            "total_timesteps": TOTAL_TIMESTEPS,
            "aggr_phy_steps": AGGR_PHY_STEPS,
            "git_commit": get_git_commit(),
            "reward_threshold": EPISODE_REWARD_THRESHOLD,
            "n_steps": args.n_steps
        },
        tags=[args.env, algo, args.obs.value, args.act.value]
    )

    # Definisce le metriche step: versioni clippate come asse X
    wandb.define_metric("train/x_timesteps")
    wandb.define_metric("eval/x_timesteps")
    wandb.define_metric("train/*", step_metric="train/x_timesteps")
    wandb.define_metric("eval/*", step_metric="eval/x_timesteps")

    return run_name

# Addestra un singolo algoritmo e restituisce risultati(best reward, tempi, path).
def train_algorithm(args, algo: str, save_dir: str) -> Dict[str, Any]:

    print(f"\n{'='*50}")
    print(f"Training {algo.upper()} on {args.env}")
    print(f"{'='*50}")

    # Setup wandb
    run_name = setup_wandb(args, algo)

    # setup ambiente
    env_name = f"{args.env}-aviary-v0"
    sa_env_kwargs = dict(
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=args.obs,
        act=args.act
    )

    # Ambiente di addestramento
    train_env = create_environment(
        env_name, sa_env_kwargs, n_envs=args.cpu,
        seed=(args.seed if args.seed >= 0 else None)
    )
    print(f"[INFO] Action space: {train_env.action_space}")
    print(f"[INFO] Observation space: {train_env.observation_space}")

    # modello
    model = create_model(algo, args.obs.value, train_env, vars(args))

    # Ambiente di valutazione
    if args.obs == ObservationType.KIN:
        eval_env = gym.make(env_name, **sa_env_kwargs)
        if args.seed >= 0:
            try:
                eval_env.reset(seed=args.seed + 123)
            except TypeError:
                eval_env.seed(args.seed + 123)
            try:
                eval_env.action_space.seed(args.seed + 124)
                eval_env.observation_space.seed(args.seed + 125)
            except Exception:
                pass
        eval_env = Monitor(eval_env)
    else:
        eval_env = create_environment(
            env_name, sa_env_kwargs, n_envs=1,
            seed=(args.seed + 123 if args.seed >= 0 else None)
        )
        eval_env = VecTransposeImage(eval_env)

    # Setup callbacks
    wandb_callback = WandbCallback(total_timesteps=TOTAL_TIMESTEPS)
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=EPISODE_REWARD_THRESHOLD,
        verbose=1
    )
    eval_callback = WandbEvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=int(EVAL_FREQ_DIVISOR / args.cpu),
        deterministic=True,
        render=False,
        total_timesteps=TOTAL_TIMESTEPS
    )

    # Addestra modello
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[wandb_callback, eval_callback],
        log_interval=100
    )
    training_time = time.time() - start_time

    # Salva il modello
    model.save(os.path.join(save_dir, 'final_model.zip'))

    # Log metriche finali (step al limite superiore per coerenza)
    wandb.log({
        "training_time_seconds": training_time,
        "final_timesteps": TOTAL_TIMESTEPS
    }, step=TOTAL_TIMESTEPS)

    # Risultati finali
    final_results = {
        "algorithm": algo,
        "run_name": run_name,
        "training_time": training_time,
        "best_reward": eval_callback.best_mean_reward,
        "save_path": save_dir
    }

    print(f"\n[INFO] {algo.upper()} training completed!")
    print(f"[INFO] Best mean reward: {eval_callback.best_mean_reward:.4f}")
    print(f"[INFO] Training time: {training_time:.2f} seconds")
    print(f"[INFO] Model saved to: {save_dir}")

    wandb.finish()
    return final_results

# Controlla coerenza degli argomenti CLI e avvisa di limitazioni 
def validate_args(args):
    if args.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        if args.env not in ['takeoff', 'hover']:
            raise ValueError("[ERROR] 1D action space is only compatible with Takeoff and HoverAviary")
        print("\n[WARNING] Using simplified 1D problem for debugging purposes\n")

    if args.cpu > 1:
        print("\n[WARNING] SAC does not support multiple environments, will use cpu=1 for SAC\n")

# Se compare_all: esegue {ppo, sac, a2c} in sequenza con lo stesso seed
# Il seed <0 indica riproducibilità disabilitata
# Cartella di salvataggio include parametri chiave e timestamp per evitare collisioni
def main():
    parser = argparse.ArgumentParser(
        description='Single agent reinforcement learning with wandb integration'
    )
    parser.add_argument(
        '--env', default='hover', type=str,
        choices=['takeoff', 'hover'],
        help='Environment to use (default: hover)'
    )
    parser.add_argument(
        '--algo', default=None, type=str,
        choices=['a2c', 'ppo', 'sac'],
        help='Algorithm to use (default: None, will train all if compare_all is True)'
    )
    parser.add_argument(
        '--obs', default='kin', type=ObservationType,
        help='Observation type (default: kin)'
    )
    parser.add_argument(
        '--act', default='one_d_rpm', type=ActionType,
        help='Action type (default: one_d_rpm)'
    )
    parser.add_argument(
        '--cpu', default=1, type=int,
        help='Number of parallel environments (default: 1)'
    )
    parser.add_argument(
        '--project', default='drone-rl-comparison', type=str,
        help='Wandb project name (default: drone-rl-comparison)'
    )
    parser.add_argument(
        '--compare_all', action='store_true',
        help='Train and compare all three algorithms (PPO, SAC, A2C)'
    )
    parser.add_argument(
        '--seed', default=-1, type=int,
        help='Seed; -1 = no seed'
    )
    parser.add_argument(
        '--n_steps', default=None, type=int,
        help='(On-policy) Rollout length per env. Scegli un divisore di TOTAL_TIMESTEPS/CPU per evitare overshoot (es. 500 con cpu=1).'
    )

    args = parser.parse_args()

    validate_args(args)

    if args.seed is None:
        args.seed = -1
    if args.seed >= 0:
        set_random_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

    if args.compare_all:
        algorithms = ['ppo', 'sac', 'a2c']
        print(f"\n[INFO] Comparing all algorithms: {algorithms}")
    elif args.algo:
        algorithms = [args.algo]
        print(f"\n[INFO] Training single algorithm: {args.algo}")
    else:
        raise ValueError("Must specify either --algo or --compare_all")

    results = []

    for algo in algorithms:
        current_cpu = 1 if algo == 'sac' else args.cpu
        current_args = argparse.Namespace(**vars(args))
        current_args.cpu = current_cpu

        save_dir = os.path.join(
            base_dir, 'results',
            f'save-{args.env}-{algo}-{args.obs.value}-{args.act.value}-{timestamp}'
        )
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "git_commit.txt"), "w") as f:
            f.write(get_git_commit() + "\n")

        try:
            result = train_algorithm(current_args, algo, save_dir)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to train {algo}: {str(e)}")
            continue

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")

        results.sort(key=lambda x: x['best_reward'], reverse=True)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['algorithm'].upper()}")
            print(f"   Best Reward: {result['best_reward']:.4f}")
            print(f"   Training Time: {result['training_time']:.2f}s")
            print(f"   Model Path: {result['save_path']}")
            print()

        print("Check your wandb dashboard for detailed comparisons!")
        print(f"Project: {args.project}")

    print(f"\n[INFO] Training completed! Check wandb project: {args.project}")


if __name__ == "__main__":
    main()
