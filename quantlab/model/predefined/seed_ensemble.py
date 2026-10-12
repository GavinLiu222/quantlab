"""Seed ensembles: one model config trained under several random seeds.

``SeedEnsemble(model, seeds)`` builds one member per seed, each a new
instance of the model's class on the model's config with ``random_seed``
replaced. The members read the same data, so they share one collected panel
and one feature request per prediction window; their predictions are
averaged after a per-bar cross-sectional z-score (``average_predictions``).
The ensemble satisfies the backtester's ``Predictor`` protocol, so it is
backtested, loaded and rebuilt from a run's ``config.json`` like a single
model.
"""

import dataclasses
from numbers import Integral
from typing import TYPE_CHECKING, Self

import xarray as xr

from quantlab.core.component import component
from quantlab.model.ensemble import BaseEnsemble

if TYPE_CHECKING:  # type hints only
    from quantlab.model.base import BaseModel


@dataclasses.dataclass(frozen=True)
class SeedEnsembleConfig:
    """What a ``SeedEnsemble`` is rebuilt from: the wrapped model and the seeds.

    Examples
    --------
    >>> SeedEnsemble(model, [0, 1]).config.seeds
    [0, 1]
    """

    #: The model every member is a copy of, under its own seed.
    model: "BaseModel" = component()
    #: The members' random seeds, in member order.
    seeds: list[int]


class SeedEnsemble(BaseEnsemble):
    """An ensemble of one model trained under several random seeds.

    Member k is ``type(model)`` built on ``model.config`` with
    ``random_seed=seeds[k]``. ``collect()`` collects once, on the first
    member, and the other members share its data backend, so the panel is
    held once. ``predict_window`` requests features once, through the first
    member (its warm-up included), and hands them to every member's
    ``predict_panel``. ``train()``, ``load()`` and ``check_checkpoint()``
    work on an ensemble unit (see ``BaseEnsemble``) whose ``run.json``
    records each member's seed; ``train_cv()`` writes one such unit per
    walk-forward fold, inside a walk-forward unit that a backtester's
    ``run_cv`` replays.

    Parameters
    ----------
    model : BaseModel
        The model to replicate. It is not trained itself; its config is the
        template of every member.
    seeds : sequence of int
        At least two distinct seeds, one per member, in member order.

    Attributes
    ----------
    model : BaseModel
        The model given, whose config ``get_config`` records.
    seeds : tuple[int, ...]
        The members' seeds, in member order.
    members : list[BaseModel]
        One model per seed.

    Raises
    ------
    TypeError
        If a seed is not an integer.
    ValueError
        If fewer than two seeds are given or a seed repeats.

    Examples
    --------
    Given a model ``model`` over one factor and one forward-return label::

        >>> ensemble = SeedEnsemble(model, [0, 1, 2])
        >>> [member.config.random_seed for member in ensemble.members]
        [0, 1, 2]
        >>> checkpoint = ensemble.collect().train()
        >>> out = ensemble.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars), out.sizes["timestamp"]
        (['fwd_ret_1'], 21)
    """

    #: The config dataclass the ensemble is serialised and rebuilt with.
    config_cls = SeedEnsembleConfig

    def __init__(self, model, seeds):
        """Initialize the ensemble; see the class docstring for parameters."""
        self.seeds = self._checked_seeds(seeds)
        self.model = model
        super().__init__(
            [
                type(model)(dataclasses.replace(model.config, random_seed=seed))
                for seed in self.seeds
            ]
        )

    @staticmethod
    def _checked_seeds(seeds) -> tuple[int, ...]:
        """Return ``seeds`` as a tuple of ints after checking count and uniqueness."""
        seeds = list(seeds)
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError(
                    f"SeedEnsemble seeds must be integers, got {seed!r} "
                    f"({type(seed).__name__})"
                )
        seeds = [int(seed) for seed in seeds]
        if len(seeds) < 2:
            raise ValueError(
                f"SeedEnsemble needs at least two seeds, got {seeds}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"SeedEnsemble seeds must be distinct, got {seeds}")
        return tuple(seeds)

    def _member_seed(self, k: int) -> int:
        """The seed of member ``k``, recorded in ``run.json``."""
        return self.seeds[k]

    def _collect(self) -> None:
        """Collect the panel once and share it with every member.

        The first member reads (see ``BaseModel._collect``); every other
        member takes that same collected panel (``use_panel``), so the record
        ``collect()`` writes holds one read and every array built from the
        panel is built once for all the members.

        Examples
        --------
        >>> ensemble.collect().members[1].collected_panel is ensemble.members[0].collected_panel
        True
        """
        first = self.members[0]
        first._collect()
        for member in self.members[1:]:
            member.use_panel(first.collected_panel)

    def _member_predictions(self, start, end) -> list[xr.Dataset]:
        """Request the features once and predict them with every member."""
        features = self.members[0]._collect_all_features(start, end)
        return [
            member.predict_panel(features).sel(timestamp=slice(start, end))
            for member in self.members
        ]

    def _member_panel_predictions(self) -> list[xr.Dataset]:
        """Predict the one shared collected panel with every member, from its one feature array."""
        return [member.predict_scored() for member in self.members]

    @property
    def config(self) -> SeedEnsembleConfig:
        """The wrapped model and the seeds; ``get_config()`` serialises it.

        Examples
        --------
        >>> config = SeedEnsemble(model, [0, 1]).get_config()
        >>> config["name"], config["seeds"], config["model"]["name"]
        ('quantlab.model.predefined.seed_ensemble.SeedEnsemble', [0, 1], 'tests.backtest_fixtures.SeededHead')
        """
        return SeedEnsembleConfig(model=self.model, seeds=list(self.seeds))

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild a seed ensemble from the dict ``get_config()`` returned.

        The wrapped model is rebuilt by the component rule, and the seeds are
        applied to it.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned, for example read back from a
            backtest run's ``config.json``.
        run_dir : str or os.PathLike, optional
            The run directory the config was read from.

        Returns
        -------
        Self
            An untrained ensemble; call ``load`` to restore a trained unit.

        Examples
        --------
        >>> rebuilt = SeedEnsemble.from_config(ensemble.get_config())
        >>> rebuilt.seeds, type(rebuilt.members[0]).__name__
        ((0, 1, 2), 'SeededHead')
        """
        fields = cls._rebuilt_fields(config, run_dir)
        return cls(fields["model"], fields["seeds"])
