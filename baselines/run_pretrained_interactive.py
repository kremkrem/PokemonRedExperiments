import uuid
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from red_gym_env import RedGymEnv
from red_gym_env import RedGymEnvConfig
from visualize_predictions import ModelVisualization


def make_env(env_conf: RedGymEnvConfig, seed=0) -> Callable[[], RedGymEnv]:
    """
    Utility function for multiprocessed env.
    :param env_conf: (dict) various environment config parameters
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        return RedGymEnv(env_conf)
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    env_config: RedGymEnvConfig = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
            }
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    file_name = 'session_4da05e87_main_good/poke_439746560_steps'
    
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    cnt = 0
    vis = ModelVisualization(model.policy)
        
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = 7 # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        last_obs = obs
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if cnt % 50 == 0:
            vis.explain_and_save_pic(last_obs, Path(file_name).parent / f'expl_{cnt}.png')
            print("Frame #" + str(cnt) + ",", action, ModelVisualization.IDX_TO_CLASS[action], vis.batch_predict(last_obs))

        cnt += 1
        if truncated:
            break
    env.close()
