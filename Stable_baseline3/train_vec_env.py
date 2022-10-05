import gym_electric_motor as gem
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from gym.wrappers import FlattenObservation
from EnvWrap4 import EnvWrap
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_electric_motor.reference_generators import SwitchedReferenceGenerator, SinusoidalReferenceGenerator, \
    WienerProcessReferenceGenerator, TriangularReferenceGenerator, StepReferenceGenerator
import numpy as np

timestamp = time.strftime("%Y%m%d-%H%M%S")
ref_gen = SwitchedReferenceGenerator([SinusoidalReferenceGenerator(amplitude_range=(0, 0.8),
                                                                   offset_range=(0, 0.5),
                                                                   frequency_range=(0.1, 5),
                                                                   reference_state='torque'),
                                      TriangularReferenceGenerator(amplitude_range=(0, 0.8),
                                                                   offset_range=(0, 0.5),
                                                                   frequency_range=(0.1, 5),
                                                                   reference_state='torque'),
                                      StepReferenceGenerator(amplitude_range=(0, 0.8),
                                                             offset_range=(0, 0.5),
                                                             frequency_range=(0.1, 5),
                                                             reference_state='torque'),
                                      WienerProcessReferenceGenerator(sigma_range=(1e-4, 1e-2),
                                                                      reference_state='torque')
                                      ],
                                     p=(0.3, 0.2, 0.3, 0.2), super_episode_length=(1000, 2000))

env_id = 'AbcCont-TC-SCIM-v0'


# env = gem.make(env_id, reference_generator=ref_gen)
# state, ref = env.reset()
#
# env = FlattenObservation(EnvWrap(env, train=True, max_episode_length=10_000))

def make_env(env_name, rank, seed=0):
    def _init():
        env = gem.make(env_name, reference_generator=ref_gen)
        env = FlattenObservation(EnvWrap(env, train=True, max_episode_length=10_000))
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def lr(start, end):
    def _lr(k):
        return start - (start - end) * (1 - k)

    return _lr


if __name__ == '__main__':
    learning_rate = 0.00025
    n_epochs = 70
    total_timesteps = 100000
    num_cpu = 8
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO(MlpPolicy,
                env=env,
                policy_kwargs=dict(net_arch=[dict(vf=[64, 64], pi=[64, 64])]),
                verbose=1,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=128
                )

    model.learn(total_timesteps=total_timesteps,
                log_interval=1)
    model.save(timestamp + "lr{:4}-epoch{:4}-{:4}".format(learning_rate, n_epochs, total_timesteps))
    print("model saved")
