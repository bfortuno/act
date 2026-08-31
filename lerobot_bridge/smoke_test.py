"""Loop 0 exit test: the isolated env imports LeRobot + the shared repo modules.

cd lerobot_bridge && uv run python smoke_test.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: F401

import _shared  # noqa: F401  (must precede shared-repo imports)
import ee_transforms


def main() -> None:
    assert ee_transforms.CANONICAL_DIM == 16
    assert ee_transforms.state_dim("rot6d") == 20
    print("shared repo root:", _shared.REPO_ROOT)
    import lerobot

    print("lerobot", lerobot.__version__)
    print("Loop 0 OK")


if __name__ == "__main__":
    main()
