from gym.core import Wrapper
from gym.spaces import Tuple, Box
import numpy as np


class EnvWrap(Wrapper):
    def __init__(self, env, max_episode_length=50_000, change_speed=None, train=False):
        super(EnvWrap, self).__init__(env)
        self.gamma = 0.99
        self.max_episode_lenght = max_episode_length
        self.change_speed = change_speed
        self.train = train
        self._time_steps = 0
        self.state_idx = np.array([
            self.env.state_names.index("omega"),
            self.env.state_names.index('i_sa'),
            self.env.state_names.index('i_sb'),
            self.env.state_names.index('u_sa'),
            self.env.state_names.index('u_sb'),
        ])

        self.observation_space = Tuple((Box(np.array([-1, -1, -1, -1, -1],),
                                            np.array([1, 1, 1, 1, 1])),
                                        env.observation_space[1],))

        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))

        self.current_idx = np.array([
            self.env.state_names.index('i_sa'),
            self.env.state_names.index('i_sb'),
            self.env.state_names.index('i_sc'),
        ])
        self.voltage_idx = np.array([
            self.env.state_names.index('u_sa'),
            self.env.state_names.index('u_sb'),
            self.env.state_names.index('u_sc'),
        ])
        self.omega_idx = self.env.state_names.index('omega')
        self.torque_idx = self.env.state_names.index('torque')

        self.limits = env.physical_system.limits
        self.tau = env.physical_system.tau

        self.weight_efficiency = 0.0
        self.weight_e_t_abs = 1 - self.weight_efficiency

        self.alphabeta_to_abc_transformation = env.physical_system.alphabeta_to_abc_space
        self.abc_to_alphabeta_transformation = env.physical_system.abc_to_alphabeta_space

        self.t_max = 0
        self._reward_log = []
        self.state = None
        self.lamda = self.tau
        self.pre_torque = 0

    def step(self, action):
        action = self.alphabeta_to_abc_transformation(action)
        (state, ref), rew, done, info = self.env.step(action)
        self.state = state

        if np.abs(state[self.torque_idx]) > self.t_max:
            self.t_max = np.abs(state[self.torque_idx])

        observable_state = state[self.state_idx]
        i_s_ab = self.abc_to_alphabeta_transformation(state[self.current_idx])
        u_s_ab = self.abc_to_alphabeta_transformation(state[self.voltage_idx])

        observable_state[1:3] = i_s_ab
        observable_state[3:5] = u_s_ab

        state_den = state * self.limits
        voltages = state_den[self.voltage_idx]
        currents = state_den[self.current_idx]

        p_in = max(np.abs(np.dot(voltages, currents)), 0.01)
        p_out = np.abs(state_den[self.torque_idx] * state_den[self.omega_idx])
        efficiency = min(p_out / p_in, 1)

        rew = self._reward(state,ref, done, efficiency)
        self._reward_log.append(rew)
        if done:
            info = {"reward": sum(self._reward_log),
                    "length": len(self._reward_log)
                    }
        else:
            info = None

        if self.train:
            if self._time_steps >= self.max_episode_length or done:
                self._time_steps = 0
                self.t_max = 0
                self.change_speed()
                done = True
        if done:
            self.reset()

        return (observable_state, ref), rew, done, info

    def _reward(self, state, ref, done, efficiency):
        not_abs = (ref[0] - state[self.torque_idx])/2
        e_t_abs = np.power(1 - np.abs(not_abs), 5)

        if done:
            return -1
        else:
            rew = self.weight_e_t_abs * e_t_abs + self.weight_efficiency * efficiency

            return rew

    def reset(self, **kwargs):
        state, ref = self.env.reset()
        self._reward_log = []
        observable_state = state[self.state_idx]
        observable_state[1:3] = self.abc_to_alphabeta_transformation(state[self.current_idx])
        observable_state[3:5] = self.abc_to_alphabeta_transformation(state[self.voltage_idx])
        return observable_state, ref
