"""Hyperparameters of the Evolution Strategies outer optimizer."""

from typing import Any, Optional, Self

from core.annotations import override
from core.optimizers.config import OptimizerConfig
from core.optimizers.es.optimizer import ESOptimizer


class ESConfig(OptimizerConfig):
    """Configuration of :class:`~core.optimizers.es.optimizer.ESOptimizer`.

    The search dimension (``dimension``) and the population size are not set
    here: ``BilevelConfig`` derives them from the mechanism space and from
    the inner optimizer's batch capacity.
    """

    def __init__(self, opt_class=None):
        super().__init__(opt_class=opt_class or ESOptimizer)

        # Add default or from default
        # ES training hyperparameters
        self.dimension: int = None
        self.sigma: float = 0.15
        self.mean_lr: float = 0.1
        self.sigma_lr: float = 0.05
        self.sigma_decay: float = 0.99
        self.min_sigma: float = 1e-3
        self.max_sigma: float = 0.5
        self.break_symmetry: bool = False
        self.convergence_eps: float = 1e-4
        self.convergence_patience: int = 10
        self.initial_mean: Optional[list[float]] = None

    @override(OptimizerConfig)
    def training(
        self,
        *,
        sigma: Optional[float] = None,
        mean_lr: Optional[float] = None,
        sigma_lr: Optional[float] = None,
        sigma_decay: Optional[float] = None,
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
        generation: Optional[int] = None,
        break_symmetry: Optional[bool] = None,
        convergence_eps: Optional[float] = None,
        convergence_patience: Optional[int] = None,
        initial_mean: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> Self:
        """Set the ES search hyperparameters. Unset arguments keep their current value.

        Parameters
        ----------
        sigma : float
            Initial standard deviation of the search distribution in logit space.
        mean_lr : float
            Step size applied to the estimated gradient when moving the mean.
        sigma_lr : float
            Strength of the sigma adaptation (``0`` disables it).
        sigma_decay : float
            Base multiplicative factor of the sigma adaptation, in ``(0, 1]``.
        min_sigma, max_sigma : float
            Bounds of sigma: a floor keeps exploring, a ceiling avoids
            destabilizing jumps.
        break_symmetry : bool
            Replace one mirrored sample by an independent one so the population
            is not strictly antithetic (allows odd population sizes).
        convergence_eps, convergence_patience : float, int
            Convergence criterion on the mean displacement.
        initial_mean : list[float], optional
            Starting point in ``[0, 1]^dimension`` (default ``0.5`` everywhere).

        Returns
        -------
        ESConfig
            ``self``, for chaining.
        """

        if sigma is not None:
            self.sigma = sigma

        if mean_lr is not None:
            self.mean_lr = mean_lr

        if sigma_lr is not None:
            self.sigma_lr = sigma_lr

        if sigma_decay is not None:
            self.sigma_decay = sigma_decay

        if min_sigma is not None:
            self.min_sigma = min_sigma

        if max_sigma is not None:
            self.max_sigma = max_sigma

        if generation is not None:
            self.generation = generation

        if break_symmetry is not None:
            self.break_symmetry = break_symmetry

        if convergence_eps is not None:
            self.convergence_eps = convergence_eps

        if convergence_patience is not None:
            self.convergence_patience = convergence_patience

        if initial_mean is not None:
            self.initial_mean = initial_mean

        return self

    # @override(OptimizerConfig)
    # def evaluation(
    #     self, *, evaluation_best_fitness, evaluation_best_candidate, **kwargs
    # ) -> Self:
    #     raise NotImplementedError

    # # TODO this is where the random seed goes
    # # TODO do we put rng here ?
    # @override(OptimizerConfig)
    # def fault_tolerance(self, *, rng: Optional[float] = None, **kwargs) -> Self:
    #     return super().fault_tolerance()
