"""The training target, computed once per bar before a fit: shared by both model variants.

``TrainingTargetMixin`` gives ``TorchModel`` and ``LibraryModel`` the one ``_transform_target``
hook and the code around it: the collected panel as a ``TrainingPanel`` of torch tensors, and
each bar's raw labels turned into the training target and its mask. It lives in the model
layer, not on ``BaseModel``, so importing the root class (``quantlab.model.base``)
never imports torch.
"""

import numpy as np
import torch

from quantlab.model.torch_data import TrainingPanel
from quantlab.utils.timer import Timer


class TrainingTargetMixin:
    """The ``_transform_target`` hook and the per-bar target computation of both variants.

    Mixed into ``TorchModel`` and ``LibraryModel`` ahead of ``BaseModel``; it reads the
    model's ``class_name``, ``to_array`` and factor and label names.
    """

    def _transform_target(self, y: torch.Tensor, training: bool):
        """Turn one bar's raw ``[S_t, L]`` labels into ``(target, keep)``.

        Shared by both variants: ``y`` is a float32 CPU tensor for a torch
        head and a library head alike, so one transform (a rank, a z-score,
        dropping the extremes) serves both.

        Called once per bar per fit, before training: ``training`` is True on
        the training bars and False on the validation and test bars. ``y``
        holds the labels of the bar's present symbols, NaN where missing.
        ``keep`` is None or ``[S_t]`` booleans; a symbol it drops leaves the
        loss but stays in the input as context. ``target`` has one row per
        symbol, or one per kept symbol. A symbol whose target is not finite
        in every label is masked out. The default returns ``(y, None)``.
        """
        return y, None

    #: The ``_transform_target`` that leaves the labels as they are. A
    #: variant that replaces the default with its own unchanged-by-default
    #: hook points this at that hook.
    _default_transform_target = _transform_target

    @property
    def label_scales(self) -> dict[str, str]:
        """Each label name's prediction scale: ``"raw"`` or ``"standardized"``.

        ``"raw"`` exactly when the model is fitted on the labels themselves
        and predicts in their units: the head keeps the default, identity
        ``_transform_target`` and, for a ``LibraryModel``, sets no
        ``training_target``. A head that overrides the hook (a rank, a
        z-score), or a library head with a ``training_target``, is
        ``"standardized"`` for every label, since the one transform applies
        to all of them.

        Examples
        --------
        >>> model.label_scales
        {'fwd_ret_1': 'raw'}
        """
        scale = "standardized" if self._standardizes_target() else "raw"
        return {str(name): scale for name in self.get_label_names()}

    def _standardizes_target(self) -> bool:
        """Whether the training target differs from the raw labels.

        True when the class overrides the variant's default ``_transform_target``.
        """
        return type(self)._transform_target is not type(self)._default_transform_target

    def _training_panel(self) -> TrainingPanel:
        """Return the collected panel as a float32 ``TrainingPanel`` with no target yet.

        Its float32 feature and label arrays are the ``collected_panel``'s,
        built once and shared by every fit on it (and by the evaluation when
        the panel is float32), so a fit reads them and never writes into them.
        """
        with Timer(f"{self.class_name}: to_array"):
            x, timestamps, symbols = self.collected_panel.array(self.get_factor_names(), np.float32)
            y_raw, _, _ = self.collected_panel.array(self.get_label_names(), np.float32)
        return TrainingPanel.from_arrays(x, timestamps=timestamps, symbols=symbols, y_raw=y_raw)

    def _fill_target(self, panel: TrainingPanel, bars, training: bool) -> None:
        """Compute the training target of ``bars`` once, through ``_transform_target``.

        Each bar's raw labels of its present symbols go through the hook
        once; the result and its validity are written into ``panel.target``
        and ``panel.mask``. ``keep`` only clears ``mask``: a dropped symbol
        stays in the feature panel as context.

        Raises
        ------
        ValueError
            If the hook returns a ``keep`` or a target of the wrong shape.
        """
        num_labels = panel.y_raw.shape[-1]
        for t in bars:
            symbols = torch.nonzero(panel.present[t]).flatten()
            n = len(symbols)
            if not n:
                continue
            target, keep = self._transform_target(panel.y_raw[t, symbols], training)
            target = torch.as_tensor(target, dtype=torch.float32).cpu()
            if keep is None:
                keep = torch.ones(n, dtype=torch.bool)
            else:
                keep = torch.as_tensor(keep, dtype=torch.bool).cpu()
                if tuple(keep.shape) != (n,):
                    raise ValueError(
                        f"{self.class_name}._transform_target: keep must have {n} "
                        f"entries at bar {panel.timestamps[t]}, got {tuple(keep.shape)}"
                    )
                if target.shape[0] == int(keep.sum()) != n:
                    full = torch.full((n, num_labels), float("nan"))
                    full[keep] = target
                    target = full
            if tuple(target.shape) != (n, num_labels):
                raise ValueError(
                    f"{self.class_name}._transform_target: expected a target of shape "
                    f"{(n, num_labels)} at bar {panel.timestamps[t]}, "
                    f"got {tuple(target.shape)}"
                )
            valid = keep & torch.isfinite(target).all(dim=-1)
            panel.target[t, symbols] = torch.where(
                valid[:, None], target, torch.zeros_like(target)
            )
            panel.mask[t, symbols] = valid
