from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
from d3rlpy.dataset import Episode, MDPDataset, Transition
from tqdm.auto import tqdm

from d3rlpy.base import LearnableBase
from d3rlpy.constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    ActionSpace,
)
from d3rlpy.iterators import RandomIterator, RoundIterator, TransitionIterator
from d3rlpy.logger import LOG


def simple_fitter(
    algo: LearnableBase,
    dataset: Union[List[Episode], List[Transition], MDPDataset],
    n_epochs: Optional[int] = None,
    n_steps: Optional[int] = None,
    n_steps_per_epoch: int = 10000,
    save_metrics: bool = True,
    experiment_name: Optional[str] = None,
    with_timestamp: bool = True,
    logdir: str = "d3rlpy_logs",
    verbose: bool = True,
    show_progress: bool = True,
    tensorboard_dir: Optional[str] = None,
    eval_episodes: Optional[List[Episode]] = None,
    save_interval: int = 1,
    scorers: Optional[Dict[str, Callable[[Any, List[Episode]], float]]] = None,
    shuffle: bool = True,
    callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """Iterate over epochs steps to train with the given dataset. At each iteration algo methods
     and properties can be changed or queried.

    .. code-block:: python

        for epoch, metrics in algo.fitter(episodes):
            my_plot(metrics)
            algo.save_model(my_path)

    Args:
        algo: d3rlpy LearnableBase already initialized algorithm.
        dataset: offline dataset to train.
        n_epochs: the number of epochs to train.
        n_steps: the number of steps to train.
        n_steps_per_epoch: the number of steps per epoch. This value will
            be ignored when ``n_steps`` is ``None``.
        save_metrics: flag to record metrics in files. If False,
            the log directory is not created and the model parameters are
            not saved during training.
        experiment_name: experiment name for logging. If not passed,
            the directory name will be `{class name}_{timestamp}`.
        with_timestamp: flag to add timestamp string to the last of
            directory name.
        logdir: root directory name to save logs.
        verbose: flag to show logged information on stdout.
        show_progress: flag to show progress bar for iterations.
        tensorboard_dir: directory to save logged information in
            tensorboard (additional to the csv data).  if ``None``, the
            directory will not be created.
        eval_episodes: list of episodes to test.
        save_interval: interval to save parameters.
        scorers: list of scorer functions used with `eval_episodes`.
        shuffle: flag to shuffle transitions on each epoch.
        callback: callable function that takes ``(algo, epoch, total_step)``
            , which is called every step.

    Returns:
        iterator yielding current epoch and metrics dict.

    """

    transitions = []
    if isinstance(dataset, MDPDataset):
        for episode in dataset.episodes:
            transitions += episode.transitions
    elif not dataset:
        raise ValueError("empty dataset is not supported.")
    elif isinstance(dataset[0], Episode):
        for episode in cast(List[Episode], dataset):
            transitions += episode.transitions
    elif isinstance(dataset[0], Transition):
        transitions = list(cast(List[Transition], dataset))
    else:
        raise ValueError(f"invalid dataset type: {type(dataset)}")

    # check action space
    if algo.get_action_type() == ActionSpace.BOTH:
        pass
    elif transitions[0].is_discrete:
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

    iterator: TransitionIterator
    if n_epochs is None and n_steps is not None:
        assert n_steps >= n_steps_per_epoch
        n_epochs = n_steps // n_steps_per_epoch
        iterator = RandomIterator(
            transitions,
            n_steps_per_epoch,
            batch_size=algo.batch_size,
            n_steps=algo.n_steps,
            gamma=algo.gamma,
            n_frames=algo.n_frames,
            real_ratio=algo._real_ratio,
            generated_maxlen=algo._generated_maxlen,
        )
        LOG.debug("RandomIterator is selected.")
    elif n_epochs is not None and n_steps is None:
        iterator = RoundIterator(
            transitions,
            batch_size=algo.batch_size,
            n_steps=algo.n_steps,
            gamma=algo.gamma,
            n_frames=algo.n_frames,
            real_ratio=algo._real_ratio,
            generated_maxlen=algo._generated_maxlen,
            shuffle=shuffle,
        )
        LOG.debug("RoundIterator is selected.")
    else:
        raise ValueError("Either of n_epochs or n_steps must be given.")

    # setup logger
    logger = algo._prepare_logger(
        save_metrics, experiment_name, with_timestamp, logdir, verbose, tensorboard_dir,
    )

    # add reference to active logger to algo class during fit
    algo._active_logger = logger

    # # initialize scaler
    # if algo._scaler:
    #     LOG.debug("Fitting scaler...", scaler=algo._scaler.get_type())
    #     algo._scaler.fit(transitions)
    #
    # # initialize action scaler
    # if algo._action_scaler:
    #     LOG.debug(
    #         "Fitting action scaler...", action_scaler=algo._action_scaler.get_type(),
    #     )
    #     algo._action_scaler.fit(transitions)
    #
    # # initialize reward scaler
    # if algo._reward_scaler:
    #     LOG.debug(
    #         "Fitting reward scaler...", reward_scaler=algo._reward_scaler.get_type(),
    #     )
    #     algo._reward_scaler.fit(transitions)

    # instantiate implementation
    if algo._impl is None:
        LOG.debug("Building models...")
        transition = iterator.transitions[0]
        action_size = transition.get_action_size()
        observation_shape = tuple(transition.get_observation_shape())
        algo.create_impl(
            algo._process_observation_shape(observation_shape), action_size
        )
        LOG.debug("Models have been built.")
    else:
        LOG.warning("Skip building models since they're already built.")

    # # save hyperparameters
    # algo.save_params(logger)
    #
    # # refresh evaluation metrics
    # algo._eval_results = defaultdict(list)

    # refresh loss history
    algo._loss_history = defaultdict(list)

    # training loop
    total_step = 0
    for epoch in range(1, n_epochs + 1):

        # dict to add incremental mean losses to epoch
        epoch_loss = defaultdict(list)

        range_gen = tqdm(
            range(len(iterator)),
            disable=not show_progress,
            desc=f"Epoch {int(epoch)}/{n_epochs}",
        )

        iterator.reset()

        for itr in range_gen:

            batch = next(iterator)
            # update parameters
            loss = algo.update(batch)

            # record metrics
            for name, val in loss.items():
                logger.add_metric(name, val)
                epoch_loss[name].append(val)

            # update progress postfix with losses
            if itr % 10 == 0:
                mean_loss = {k: np.mean(v) for k, v in epoch_loss.items()}
                range_gen.set_postfix(mean_loss)

            total_step += 1

            # call callback if given
            if callback:
                callback(algo, epoch, total_step)

        # save loss to loss history dict
        algo._loss_history["epoch"].append(epoch)
        algo._loss_history["step"].append(total_step)
        for name, vals in epoch_loss.items():
            if vals:
                algo._loss_history[name].append(np.mean(vals))

        if scorers and eval_episodes:
            algo._evaluate(eval_episodes, scorers, logger)

        # save metrics
        metrics = logger.commit(epoch, total_step)

        # save model parameters
        if epoch % save_interval == 0:
            logger.save_model(total_step, algo)

        yield epoch, metrics

    # drop reference to active logger since out of fit there is no active
    # logger
    algo._active_logger = None
