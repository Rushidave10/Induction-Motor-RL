import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import gym
from RNN_model import ActorCriticModel
from docopt import docopt
from gym.wrappers import FlattenObservation
import gym_electric_motor as gem
from EnvWrap_ab import EnvWrap
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gym_electric_motor.reference_generators import SwitchedReferenceGenerator, SinusoidalReferenceGenerator, \
    WienerProcessReferenceGenerator, TriangularReferenceGenerator, StepReferenceGenerator, ConstReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.motor_dashboard_plots.base_plots import TimePlot


def main():
    # Command line arguments via docopt
    _USAGE = """
        Usage:
            enjoy.py [options]
            enjoy.py --help

        Options:
            --model=./models/ppov9.nn              Specifies the path to the trained model [default: ./models/ppov9.nn].
        """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    class ExternalPlot(TimePlot):

        def __init__(self, referenced=False, additional_lines=0, min=0, max=1):
            """
                This function creates an object for external plots in a GEM MotorDashboard.
                Args:
                    referenced: a reference is to be displayed
                    additional_lines: number of additional lines in plot
                    min: minimum y-value of the plot
                    max: maximum y-value of the plot
                Returns:
                    Object that can be passed to a GEM environment to plot additional data.
            """
            super().__init__()

            self._state_line_config = self._default_time_line_cfg.copy()
            self._ref_line_config = self._default_time_line_cfg.copy()
            self._add_line_config = self._default_time_line_cfg.copy()

            self._referenced = referenced
            self.min = min
            self.max = max

            # matplotlib-Lines for the state and reference
            self._state_line = None
            self._reference_line = None

            self.state_label = ''
            self.ref_label = ''

            # Data containers
            self._state_data = []
            self._reference_data = []
            self._additional_data = []

            # Add additional lines
            self.added = additional_lines > 0
            self.add_lines = additional_lines

            if self.added:
                self.add_labels = []
                self._additional_lines = []
                for i in range(additional_lines):
                    self._additional_lines.append([])
                    self._additional_data.append(None)
                    self.add_labels.append('')

        def set_env(self, env):
            # Docstring of superclass
            super().set_env(env)
            self._label = None
            self._y_lim = (self.min, self.max)
            self.reset_data()

        def reset_data(self):
            # Docstring of superclass
            super().reset_data()
            # Initialize the data containers
            self._state_data = np.full(shape=self._x_data.shape, fill_value=np.nan)
            self._reference_data = np.full(shape=self._x_data.shape, fill_value=np.nan)

            if self.added:
                for i in range(self.add_lines):
                    self._additional_data[i] = np.full(shape=self._x_data.shape, fill_value=np.nan)

        def initialize(self, axis):
            # Docstring of superclass
            super().initialize(axis)

            # Line to plot the state data
            self._state_line, = self._axis.plot(self._x_data, self._state_data, **self._state_line_config,
                                                zorder=self.add_lines + 2)
            self._lines = [self._state_line]

            # If the state is referenced plot also the reference line
            if self._referenced:
                self._reference_line, = self._axis.plot(self._x_data, self._reference_data, **self._ref_line_config,
                                                        zorder=self.add_lines + 1)
                # axis.lines = axis.lines[::-1]
                self._lines.append(self._reference_line)

            self._y_data = [self._state_data, self._reference_data]

            # If there are added lines plot also these lines
            if self.added:
                for i in range(self.add_lines):
                    self._additional_lines[i], = self._axis.plot(self._x_data, self._additional_data[i],
                                                                 **self._add_line_config, zorder=self.add_lines - i)
                    self._lines.append(self._additional_lines[i])
                    self._y_data.append(self._additional_data[i])

            # Set the labels of the refernce line and additional lines
            if self._referenced:
                if self.added:
                    lines = [self._state_line, self._reference_line]
                    lines.extend(self._additional_lines)
                    labels = [self.state_label, self.ref_label]
                    labels.extend(self.add_labels)
                    self._axis.legend((lines), (labels), loc='upper left', numpoints=20)
                else:
                    self._axis.legend(([self._state_line, self._reference_line]), ([self.state_label, self.ref_label]),
                                      loc='upper left', numpoints=20)
            else:
                self._axis.legend((self._state_line,), (self.state_label,), loc='upper left', numpoints=20)

        def set_label(self, labels):
            """
                Method to set the labels, A dict must be passed. The keys are: y_label, state_label, ref_label, add_label.
                For the key add_label a list with the length of the number of additional lines is passed.
            """

            self._label = labels.get('y_label', '')
            self.state_label = labels['state_label']
            if self._referenced:
                self.ref_label = labels.get('ref_label', '')
            if 'add_label' in labels.keys():
                self.add_labels = labels['add_label']

        def on_step_end(self, k, state, reference, reward, done):
            super().on_step_end(k, state, reference, reward, done)
            idx = self.data_idx
            self._x_data[idx] = self._t

        def add_data(self, additional_data):
            """Method to pass the external data. A list must be passed with the length of the number of plots."""
            idx = self.data_idx
            # Write the data to the data containers
            if self._referenced:
                self._state_data[idx] = additional_data[0]
                self._reference_data[idx] = additional_data[1]
                if self.added:
                    for i in range(self.add_lines):
                        self._additional_data[i][idx] = additional_data[i + 2]
            elif self.added:
                self._state_data[idx] = additional_data[0]
                for i in range(self.add_lines):
                    self._additional_data[i][idx] = additional_data[i + 1]
            else:
                self._state_data[idx] = additional_data[0]

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
                                         p=(0.25, 0.25, 0.25, 0.25), super_episode_length=(1000, 10_000))

    ref_gen1 = ConstReferenceGenerator(reference_state="torque", reference_value=0.75)
    abs_plot = ExternalPlot()

    visualization = MotorDashboard(state_plots=['torque', 'i_sd', 'i_sq',
                                                'u_sd', 'u_sq'],
                                   additional_plots=(abs_plot,),
                                   )

    # Instantiate environment
    env = gem.make(config["env"],
                   reference_generator=ref_gen,
                   visualization=visualization,
                   load=ConstantSpeedLoad(1000 * np.pi / 30)
                   )
    ps = env.physical_system
    abs_plot.set_label({'y_label': 'Flux', 'state_label': r'$\Psi_{abs}$'})
    flux_idx = ps._motor_ode_idx[
               ps._electrical_motor.PSI_RALPHA_IDX:ps._electrical_motor.PSI_RBETA_IDX + 1]

    state, ref = env.reset()
    env = FlattenObservation(EnvWrap(env))
    obs = env.reset()

    # env = gym.make(config["env"])
    # obs = env.reset()
    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, env.action_space)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run and render episode
    done = False
    episode_rewards = []

    # Init recurrent cell
    hxs, cxs = model.init_recurrent_cell_states(1)
    if config["recurrence"]["layer_type"] == "gru":
        recurrent_cell = hxs
    elif config["recurrence"]["layer_type"] == "lstm":
        recurrent_cell = (hxs, cxs)
    actions = []
    obs = env.reset()
    for i in range(20_000):
        # Forward model
        policy, value, recurrent_cell = model(torch.tensor(obs), recurrent_cell, 1)
        # Sample action
        # action = policy.sample().cpu().numpy()
        action = policy.detach().numpy()
        actions.append(action)
        # Step environemnt
        obs, reward, done, info = env.step(action.squeeze(0))
        episode_rewards.append(reward)

        real_flux = env.physical_system._ode_solver.y[flux_idx]
        abs_plot.add_data([np.abs(np.complex(real_flux[0], real_flux[1]))])

        if done:
            obs = env.reset()

        env.render()
    plt.plot(actions)
    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

    plt.show(block=True)
    env.close()


if __name__ == "__main__":
    main()
