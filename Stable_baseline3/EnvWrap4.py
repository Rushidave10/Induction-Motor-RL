from gym.core import Wrapper
from gym.spaces import Tuple, Box
import numpy as np
from random import randint


class EnvWrap(Wrapper):
    def __init__(self, env, max_episode_length=50_000, train=False):
        super().__init__(env)
        self.gamma = 0.99
        self.max_episode_length = max_episode_length
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

        self.action_space = Box(np.array([-0.3, -1]), np.array([0.3, 1]))

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

        self.weight_efficicy = 0.05
        self.weight_e_t_abs = 1 - self.weight_efficicy

        self.alphabeta_to_abc_transformation = env.physical_system.alphabeta_to_abc_space
        self.abc_to_alphabeta_transformation = env.physical_system.abc_to_alphabeta_space
        self.abc_to_dq_transformation = env.physical_system.abc_to_dq_space
        self.dq_to_abc_transformation = env.physical_system.dq_to_abc_space

        self.u_alpha_integrator = 0
        self.u_beta_integrator = 0
        self.psi_angle = 0
        self.t_max = 0
        self._reward_log = 0

    def step(self, action):
        action = self.dq_to_abc_transformation(action, self.psi_angle)

        (state, ref), rew, done, info = self.env.step(action)
        self._time_steps += 1

        if np.abs(state[self.torque_idx]) > self.t_max:
            self.t_max = np.abs(state[self.torque_idx])

        observable_state = state[self.state_idx]
        u_alpha_beta = self.abc_to_alphabeta_transformation(state[self.voltage_idx])

        self.u_alpha_integrator = self.u_alpha_integrator * 0.996 + u_alpha_beta[0] * self.tau
        self.u_beta_integrator = self.u_beta_integrator * 0.996 + u_alpha_beta[1] * self.tau
        self.psi_angle = np.angle(np.complex(self.u_alpha_integrator, self.u_beta_integrator))
        psi_abs = np.abs(np.complex(self.u_alpha_integrator, self.u_beta_integrator))

        i_s_dq = self.abc_to_dq_transformation(state[self.current_idx], self.psi_angle)
        u_s_dq = self.abc_to_dq_transformation(state[self.voltage_idx], self.psi_angle)

        t_hat = psi_abs * 3 * i_s_dq[1] * self.limits[self.torque_idx] / self.limits[self.current_idx[0]]

        observable_state[1:3] = i_s_dq
        observable_state[3:5] = u_s_dq
        observable_state = np.append(observable_state, psi_abs)
        observable_state = np.append(observable_state, t_hat)

        state_den = state * self.limits
        voltages = state_den[self.voltage_idx]
        currents = state_den[self.current_idx]

        p_in = 3 / 2 * max(np.sum(np.abs(voltages * currents)), 1e-5)
        p_out = np.abs(state_den[self.torque_idx] * state_den[self.omega_idx])
        efficiency = min(p_out / p_in, 1)

        rew = self._reward(state, ref, done, efficiency)

        if self.train:
            self._reward_log += rew
            if self._time_steps >= self.max_episode_length or done:
                print(f'Omega:           {self.env.physical_system.mechanical_load._omega}\n',
                      f'Steps:           {self._time_steps}\n',
                      f'Reward:          {self._reward_log}\n',
                      f'Reward per Step: {self._reward_log / self._time_steps}\n',
                      f'Maximum Torque:  {self.t_max}\n')

                self._time_steps = 0
                self.t_max = 0
                self._reward_log = 0
                done = True

                omega = randint(0, 2500)
                self.env.physical_system.mechanical_load._omega = omega * np.pi / 30

        if done:
            self.reset()

        return (observable_state, ref), rew, done, info

    def _reward(self, state, ref, done, efficiency):
        e_t_abs = np.power(1 - np.abs(ref[0] - state[self.torque_idx]) / 2, 5)
        if done:
            return 0
        else:
            return self.weight_e_t_abs * e_t_abs + self.weight_efficicy * efficiency

    def reset(self, **kwargs):
        state, ref = self.env.reset()
        self.u_alpha_integrator = 0
        self.u_beta_integrator = 0
        self.psi_angle = 0
        observable_state = state[self.state_idx]
        observable_state[1:3] = self.abc_to_dq_transformation(state[self.current_idx], self.psi_angle)
        observable_state[3:5] = self.abc_to_dq_transformation(state[self.voltage_idx], self.psi_angle)
        observable_state = np.append(observable_state, 0)
        observable_state = np.append(observable_state, 0)
        return observable_state, ref
