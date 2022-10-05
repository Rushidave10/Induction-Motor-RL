from gym.core import Wrapper
from gym.spaces import Tuple, Box
import numpy as np


class EnvWrap(Wrapper):
    def __init__(self, env, max_episode_length=50000, change_speed=None, train=False):
        super().__init__(env)
        self.gamma = 0.99
        self.max_episode_length = max_episode_length
        self.change_speed = change_speed
        self.train = train
        self._time_steps = 0
        self.state_idx = np.array([
            self.env.state_names.index('omega'),
            self.env.state_names.index('i_sa'),
            self.env.state_names.index('i_sb'),
            self.env.state_names.index('u_sa'),
            self.env.state_names.index('u_sb'),
        ])

        self.observation_space = Tuple((Box(np.array([-1, -1, -1, -1, -1, 0, -1],),
                                            np.array([1, 1, 1, 1, 1, 1, 1])),
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
        self.abc_to_dq_transformation = env.physical_system.abc_to_dq_space
        self.dq_to_abc_transformation = env.physical_system.dq_to_abc_space

        self.u_alpha_integrator = 0
        self.u_beta_integrator = 0
        self.psi_angle = 0
        self.psi_abs = 0
        self.t_max = 0
        self._reward_log = []
        self.state = None
        self.lamda = self.tau
        self.pre_torque = 0

    def step(self, action):
        action = self.dq_to_abc_transformation(action, self.psi_angle)

        (state, ref), rew, done, info = self.env.step(action)
        self.state = state
        self._time_steps += 1

        if np.abs(state[self.torque_idx]) > self.t_max:
            self.t_max = np.abs(state[self.torque_idx])

        observable_state = state[self.state_idx]
        u_alpha_beta = self.abc_to_alphabeta_transformation((self.limits * state)[self.voltage_idx])
        i_alpha_beta = self.abc_to_alphabeta_transformation((self.limits * state)[self.current_idx])

        self.u_alpha_integrator = self.u_alpha_integrator * (1 - self.lamda) + u_alpha_beta[0] * self.lamda
        self.u_beta_integrator = self.u_beta_integrator * (1 - self.lamda) + u_alpha_beta[1] * self.lamda
        self.psi_angle = np.angle(np.complex(self.u_alpha_integrator, self.u_beta_integrator))
        psi_abs = np.abs(np.complex(self.u_alpha_integrator, self.u_beta_integrator))
        self.psi_abs = psi_abs

        i_s_dq = self.abc_to_dq_transformation(state[self.current_idx], self.psi_angle)
        u_s_dq = self.abc_to_dq_transformation(state[self.voltage_idx], self.psi_angle)

        t_hat = 3 / 2 * (self.u_alpha_integrator * i_alpha_beta[1] - self.u_beta_integrator * i_alpha_beta[0]) / self.limits[self.torque_idx]

        observable_state[1:3] = i_s_dq
        observable_state[3:5] = u_s_dq
        observable_state = np.append(observable_state, psi_abs)
        observable_state = np.append(observable_state, t_hat)

        state_den = state * self.limits
        voltages = state_den[self.voltage_idx]
        currents = state_den[self.current_idx]

        p_in = max(np.abs(np.dot(voltages, currents)), 0.01)
        p_out = np.abs(state_den[self.torque_idx] * state_den[self.omega_idx])
        efficiency = min(p_out / p_in, 1)

        rew = self._reward(state, observable_state, ref, done, efficiency)
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

    def _reward(self, state, observable_state, ref, done, efficiency):
        not_abs = (ref[0] - state[self.torque_idx])/2
        e_t_abs = np.power(1 - np.abs(not_abs), 4)
        flux_abs = observable_state[5]

        if done:
            return -1
        else:
            rew = self.weight_e_t_abs * e_t_abs + self.weight_efficiency * efficiency
            # if flux_abs <= 0.1:
            #     rew = rew - 0.1
            return rew

    def reset(self, **kwargs):
        state, ref = self.env.reset()
        self.u_alpha_integrator = 0
        self.u_beta_integrator = 0
        self.psi_angle = 0
        self.psi_abs = 0
        self._reward_log = []
        observable_state = state[self.state_idx]
        observable_state[1:3] = self.abc_to_dq_transformation(state[self.current_idx], self.psi_angle)
        observable_state[3:5] = self.abc_to_dq_transformation(state[self.voltage_idx], self.psi_angle)
        observable_state = np.append(observable_state, 0)
        observable_state = np.append(observable_state, 0)
        return observable_state, ref
