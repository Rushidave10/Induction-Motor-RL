import torch
from config import scim
from trainer import PPOTrainer
import gym_electric_motor as gem
import numpy as np
from gym.wrappers import FlattenObservation
from Envwrap_dq import EnvWrap
from gym_electric_motor.reference_generators import SwitchedReferenceGenerator, SinusoidalReferenceGenerator, \
    WienerProcessReferenceGenerator, TriangularReferenceGenerator, StepReferenceGenerator, ConstReferenceGenerator

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
                                                                      reference_state='torque'),
                                      ],
                                     p=(0.25, 0.25, 0.25, 0.25), super_episode_length=(3000, 4000))


def speed_load(minimum, maximum):
    speed = np.random.randint(minimum, maximum) * np.pi / 30

    def _speed_load(t):
        nonlocal speed
        return speed

    def _change_speed():
        nonlocal speed
        speed = np.random.randint(minimum, maximum) * np.pi / 30

    return _speed_load, _change_speed


speed, change_speed = speed_load(0, 2500)
ref_gen1 = ConstReferenceGenerator(reference_state="torque", reference_value=0.5)
env_id = 'AbcCont-TC-SCIM-v0'
env = gem.make(env_id, reference_generator=ref_gen)
state, ref = env.reset()
env = FlattenObservation(EnvWrap(env, train=True, max_episode_length=10000, change_speed=change_speed))

torch.set_default_tensor_type("torch.FloatTensor")

trainer = PPOTrainer(scim())
trainer.run_training()
trainer.close()
