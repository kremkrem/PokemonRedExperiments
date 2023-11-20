from os.path import exists
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from argparse_pokemon import *
from custom_networks import linear_schedule
from red_gym_env import StackedRedGymEnv
from torch.nn.parameter import Parameter


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = StackedRedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


def multiply(l: Parameter):
    res = 1
    for d in l.size():
        res *= d
    return res


if __name__ == '__main__':

    ep_length = 2048 * 8
    sess_path = f'session_{str(uuid.uuid4())[:8]}'
    print("Starting session ", sess_path)
    args = get_args('run_baseline_parallel.py',
                    ep_length=ep_length,
                    sess_path=sess_path)

    env_config = {
        'headless': True,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'action_freq_random_offset': 1,
        'init_state': '../has_pokedex_nballs.state',
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': True,
        'fast_video': True,
        'full_video': True,
        'move_list_zoom': False,
        'session_path': Path(sess_path),
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'sim_frame_dist': 2_000_000.0
    }
    env_config_alt = env_config.copy()
    env_config_alt['save_video'] = False

    #env_config = change_env(env_config, args)

    num_cpu = 4  #44 #64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([
        make_env(i, env_config if i == 0 else env_config_alt)
        for i in range(num_cpu)
    ])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length,
                                             save_path=sess_path,
                                             name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 100
    file_name = 'bogus'

    model: Optional[PPO] = None
    #'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('\nNo checkpoint found, creating new model')
        model = PPO('CnnPolicy',
                    env,
                    verbose=1,
                    n_steps=ep_length,
                    batch_size=512,
                    n_epochs=1,
                    gamma=0.999,
                    learning_rate=linear_schedule(0.0005, 0.0002))

    print(model.policy)
    print("Total no. params:")
    print(sum([multiply(parameter)
               for parameter in model.policy.parameters()]))

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * num_cpu * 1000,
                    callback=checkpoint_callback)
