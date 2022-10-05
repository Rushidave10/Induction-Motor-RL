import gym
import numpy as np
import os
import pickle
import torch
import time
from torch import optim
from buffer import Buffer
from RNN_model import ActorCriticModel
from utility import polynomial_decay
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import gym_electric_motor as gem
from EnvWrap_ab import EnvWrap
from gym.wrappers import FlattenObservation


class PPOTrainer:

    def __init__(self,
                 config: dict) -> None:

        self.config = config
        self.recurrence = config["recurrence"]
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]

        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter(
            "./summaries" + "/sequence_len{:4}-updates{:4}-worker_steps{:4}-".format(
                self.config["recurrence"]["sequence_len"],
                self.config["updates"],
                self.config["worker_steps"]) + timestamp)

        print("Step 1: Init dummy environment")
        dummy_env = gem.make(self.config["env"])
        dummy_env = FlattenObservation(EnvWrap(dummy_env, max_episode_length=10_000))
        observation_space = dummy_env.observation_space
        action_space = dummy_env.action_space
        dummy_env.close()

        print("Step 2: Init Buffer")
        self.buffer = Buffer(self.config, observation_space, action_space)

        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, observation_space, action_space)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        print("Step 4: Init environment")
        self.obs = np.zeros((1,) + observation_space.shape, dtype=np.float32)
        print(f'init obs shape:{self.obs.shape}')

        hxs, cxs = self.model.init_recurrent_cell_states(num_sequences=1)

        if self.recurrence["layer_type"] == "gru":
            self.recurrence_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrence_cell = (hxs, cxs)

        print("Step 5: Reset Environment")
        self.env = gem.make(self.config["env"])
        state, ref = self.env.reset()
        self.env = FlattenObservation(EnvWrap(self.env, max_episode_length=10_000))
        self.obs = self.env.reset()

    def run_training(self) -> None:

        print("Step 6: Starting the training")
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            #  Decay the hyper-parameter polynomially based on the config.
            learning_rate = polynomial_decay(self.lr_schedule["initial"],
                                             self.lr_schedule["final"],
                                             self.lr_schedule["max_decay_steps"],
                                             self.lr_schedule["power"],
                                             update)
            beta = polynomial_decay(self.beta_schedule["initial"],
                                    self.beta_schedule["final"],
                                    self.beta_schedule["max_decay_steps"],
                                    self.beta_schedule["power"],
                                    update)
            clip_range = polynomial_decay(self.cr_schedule["initial"],
                                          self.cr_schedule["final"],
                                          self.cr_schedule["max_decay_steps"],
                                          self.cr_schedule["power"],
                                          update)

            sampled_episode_info = self._sample_training_data()

            self.buffer.prepare_batch_dict()

            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_infos(episode_infos)

            result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f}" \
                     " v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} " \
                     "advantage={:.3f}".format(update,
                                               episode_result["reward_mean"],
                                               episode_result["reward_std"],
                                               episode_result["length_mean"],
                                               episode_result["length_std"],
                                               training_stats[0],
                                               training_stats[1],
                                               training_stats[3],
                                               training_stats[2],
                                               torch.mean(self.buffer.values),
                                               torch.mean(self.buffer.advantages)
                                               )
            print(result)
            self._write_training_summary(update, training_stats, episode_result)

        #  Save the trained model at the end of the training
        self._save_model()

    def _sample_training_data(self) -> list:

        episode_info = []
        for t in range(self.config["worker_steps"]):
            with torch.no_grad():
                self.buffer.obs[t] = torch.tensor(self.obs)
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[t] = self.recurrence[0].squeeze(0)
                elif self.recurrence["layer_type"] == "lstm":
                    self.buffer.hxs[t] = self.recurrence_cell[0].squeeze(0)
                    self.buffer.cxs[t] = self.recurrence_cell[1].squeeze(0)

                policy, value, self.recurrence_cell = self.model(torch.tensor(self.obs),
                                                                 self.recurrence_cell,
                                                                 )

                self.buffer.values[t] = value
                action = policy.sample()
                log_prob = policy.log_prob(action)
                self.buffer.actions[t] = action
                self.buffer.log_probs[t] = log_prob

            obs, self.buffer.rewards[t], self.buffer.dones[t], info = self.env.step(self.buffer.actions[t])
            if info:
                episode_info.append(info)
                obs = self.env.reset()
                if self.recurrence["reset_hidden_state"]:
                    hxs, cxs = self.model.init_recurrent_cell_states(1)
                    if self.recurrence["layer_type"] == "gru":
                        self.recurrence_cell[:] = hxs
                    elif self.recurrence["layer_type"] == "lstm":
                        self.recurrence_cell[0][:] = hxs
                        self.recurrence_cell[1][:] = cxs
            self.obs = obs

        _, last_value, _ = self.model(torch.tensor(self.obs), self.recurrence_cell)
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_info

    def _train_epochs(self, learning_rate: float, clip_range: float, beta: float) -> list:
        train_info = []
        for _ in range(self.config["epochs"]):
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
        return train_info

    def _train_mini_batch(self, samples: dict, learning_rate: float, clip_range: float, beta: float):

        #  Retrieve sampled recurrent cell states to feed the model.
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        #  Forward Model
        policy, value, _ = self.model(samples["obs"], recurrent_cell, self.recurrence["sequence_len"])

        #  Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (
                samples["advantages"].std() + 1e-8)  # To avoid division by zero.

        log_probs = policy.log_prob(samples["actions"])
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr_1 = ratio * normalized_advantage
        surr_2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr_1, surr_2)
        policy_loss = PPOTrainer._masked_mean(policy_loss, samples["loss_mask"])

        #  Value function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = PPOTrainer._masked_mean(vf_loss, samples["loss_mask"])

        entropy_bonus = PPOTrainer._masked_mean(policy.entropy(), samples["loss_mask"])

        #  Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        #  Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return [policy_loss.data.numpy(),
                vf_loss.data.numpy(),
                loss.data.numpy(),
                entropy_bonus.data.numpy(),
                # self.model.recurrent_layer.weight_ih_l0
                ]

    def _write_training_summary(self, update, training_stats, episode_result, ) -> None:
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[2], update)
        self.writer.add_scalar("losses/sequence_length", self.buffer.true_sequence_len, update)
        self.writer.add_scalar("losses/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("losses/advantage_mean", torch.mean(self.buffer.advantages), update)
        self.writer.add_histogram("LSTM_weight_ih", self.model.recurrent_layer.weight_ih_l0, update)
        self.writer.add_histogram("LSTM_weight_hh", self.model.recurrent_layer.weight_hh_l0, update)
        self.writer.add_histogram("Output_layer_Actor", self.model.policy.weight, update)
        self.writer.add_histogram("Output_layer_Critic", self.model.value.weight, update)

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

    @staticmethod
    def _process_episode_infos(episode_info: deque) -> dict:
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def _save_model(self) -> None:
        """
        Saves the model and the used training config to the models directory. The filename is based on run_id.
        """
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + "ppov9.nn", "wb"))
        print("Model saved to " + "./models/ " + "ppov9.nn")

    def close(self) -> None:
        """
        Terminates the trainer and all related processes.
        """
        self.env.close()
        self.writer.close()
        time.sleep(1.0)
        exit(0)

