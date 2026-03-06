"""
Deprecated – use the unified ``train.py`` instead::

    torchrun --nproc_per_node=N src/runs/train.py model=mvit

This wrapper remains for backward compatibility and simply forwards to
the unified entry-point.
"""

import warnings
warnings.warn(
    "train_mvit.py is deprecated. "
    "Use: torchrun ... src/runs/train.py model=mvit",
    DeprecationWarning,
    stacklevel=1,
)

from runs.train import main   # noqa: E402

if __name__ == "__main__":
    main()

