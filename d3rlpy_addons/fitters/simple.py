import numpy as np

from tqdm.auto import tqdm
from d3rlpy.preprocessing.stack import StackedObservation
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.logger import D3RLPyLogger
from d3rlpy.dataset import MDPDataset
from d3rlpy.online.utility import get_action_size_from_env


class EpisodicFitter:
    def __init__(self,
                 env,
                 algo,
                 buffer,
                 explorer=None,
                 act_fn=lambda action: action,
                 obs_fn=lambda observation: observation,
                 n_steps=1000000,
                 n_steps_per_epoch=10000,
                 update_interval=1,
                 update_start_step=0,
                 save_metrics=False,
                 experiment_name=None,
                 with_timestamp=False,
                 logdir='d3rlpy_logs',
                 verbose=False,
                 tensorboard=False):
        """
        Simplified online trainer generator.

        Args:
            env (gym.Env): gym-like environment.
            algo (d3rlpy.algos.base.AlgoBase): algorithm.
            buffer (d3rlpy.online.buffers.Buffer): replay buffer.
            explorer (d3rlpy.online.explorers.Explorer): action explorer.
            act_fn: a callable to transform action from algo before sendint it to env.
            obs_fn: a callable to transform osbervations from env before algo.get_action.
            n_steps (int): the number of total steps to train.
            n_steps_per_epoch (int): the number of steps per epoch.
            update_interval (int): the number of steps per update.
            update_start_step (int): the steps before starting updates.
            save_metrics (bool): flag to record metrics. If False, the log
                directory is not created and the model parameters are not saved.
            experiment_name (str): experiment name for logging. If not passed,
                the directory name will be `{class name}_online_{timestamp}`.
            with_timestamp (bool): flag to add timestamp string to the last of
                directory name.
            logdir (str): root directory name to save logs.
            verbose (bool): flag to show logged information on stdout.
            tensorboard (bool): flag to save logged information in tensorboard
                (additional to the csv data)

        """
        self.env = env
        self.algo = algo
        self.buffer = buffer
        self.explorer = explorer
        self.act_fn = act_fn
        self.obs_fn = obs_fn
        self.n_steps = n_steps
        self.n_steps_per_epoch = n_steps_per_epoch
        self.update_interval = update_interval
        self.update_start_step = update_start_step

        # setup logger
        if experiment_name is None:
            experiment_name = algo.__class__.__name__ + '_online'

        self.logger = D3RLPyLogger(experiment_name,
                                   save_metrics=save_metrics,
                                   root_dir=logdir,
                                   verbose=verbose,
                                   tensorboard=tensorboard,
                                   with_timestamp=with_timestamp)

        # setup algorithm
        if self.algo.impl is None:
            self.algo.build_with_env(env)

        # save hyperparameters
        self.algo._save_params(self.logger)
        self.batch_size = self.algo.batch_size

    def fitter(self, max_steps, max_episodes=None, max_steps_per_episode=None):
        """
        Perform an environment step.

        :param max_steps: maximum number of steps to run
        :param max_episodes: maximum number of episodes allowed to run
        :param max_steps_per_episode: maximum number of steps allowed per episode
        :return: an environment step as:
            metrics: a dict with metrics collected from all the steps
            (observation: the observation after env step ,
            reward: the reward after env step,
            terminal: terminal flag after env step,
            info: info dict returned by env step)
        """
        if max_steps_per_episode is None:
            max_steps_per_episode = max_steps
        if max_episodes is None:
            max_episodes = max_steps

        # start training loop
        metrics = {}
        episode = 0
        terminal = True
        episode_step = 1
        reward = 0

        for total_step in range(max_steps):
            if terminal or episode_step > max_steps_per_episode:
                episode += 1
                episode_step = 1
                observation = self.env.reset()
                reward, terminal, info = 0.0, False, {}
                if episode > max_episodes:
                    break

            # sample exploration action
            observation = self.obs_fn(observation)
            action = self.get_action(observation, total_step)
            action = self.act_fn(action)

            # store observation in buffer
            self.buffer.append(observation, action, reward, terminal)

            # get next observation
            observation, reward, terminal, info = self.env.step(action)

            # pseudo epoch count
            epoch = total_step // self.n_steps_per_epoch

            is_epoch_update = (total_step > self.update_start_step
                               and len(self.buffer) > self.batch_size
                               and total_step % self.update_interval)

            if is_epoch_update:
                # sample mini-batch
                batch = self.buffer.sample(batch_size=self.batch_size,
                                           n_frames=self.algo.n_frames,
                                           n_steps=self.algo.n_steps,
                                           gamma=self.algo.gamma)

                # update algo parameters
                loss = self.algo.update(epoch, total_step, batch)

                # record metrics
                for name, val in zip(self.algo._get_loss_labels(), loss):
                    if val:
                        self.logger.add_metric(name, val)

            if total_step % self.n_steps_per_epoch == 0:

                # until new logger version return metrics
                for name, buffer in self.logger.metrics_buffer.items():
                    metric = np.mean(buffer)
                    metrics[name] = metric

                # save new metrics
                self.logger.commit(epoch, total_step)

            episode_step += 1
            metrics["episode"] = episode
            metrics["episode_step"] = episode_step
            metrics["total_steps"] = total_step
            metrics["total_steps"] = total_step

            yield metrics, (observation, reward, terminal, info)

    def get_action(self, obs, step):
        if self.explorer:
            act = self.explorer.sample(self.algo, obs, step)
        else:
            act = self.algo.sample_action([obs])[0]
        return act
