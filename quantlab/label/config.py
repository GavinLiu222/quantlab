"""Configuration for using future factor values as prediction targets.

``ForwardConfig`` selects a factor and the number of output bars to shift.
Pass it to ``Forward`` to create a training label.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.factor.base import Factor


@dataclass(kw_only=True, frozen=True)
class ForwardConfig(FrozenConfig):
    """Configure the factor and future-bar offset used by a ``Forward`` label.

    The label at bar ``t`` reads the factor at ``t + delay + span``.
    ``span`` must be at least one and ``delay`` must be nonnegative. The
    wrapped factor supplies the input data and optional saved store.

    Examples
    --------
    With ``factor`` a factor built earlier:

    >>> cfg = ForwardConfig(factor=factor, span=5)
    >>> cfg.span, cfg.delay
    (5, 1)
    """

    #: The factor shifted forward to make the label.
    factor: "Factor" = component()
    #: Bars the label accumulates over, such as the n bars of an n-bar
    #: forward return.
    span: int
    #: Bars between the bar a signal forms on and the first bar the label
    #: counts; 1 because a signal at t fills at t+1's open.
    delay: int = 1
    #: Dotted import path of the label class; filled by the config setter.
    name: str | None = None
