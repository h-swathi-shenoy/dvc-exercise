import typing as t
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PathConfig:
    base_path: t.Optional[Path] = Path(__file__).absolute().parent.parent.parent
    data_dir: t.Optional[Path] = None
    configs_dir: t.Optional[Path] = None
    models_dir: t.Optional[Path] = None

    def __post_init__(self):
        self.data_dir = self.base_path.joinpath('data')
        self.configs_dir = self.base_path.joinpath('configs')
        self.models_dir = self.base_path.joinpath('models')

    def to_dict(self):
        path = asdict(self)
        paths = {k: str(v) for k, v in path.items()}
        return paths


if __name__ == '__main__':
    path_repo = PathConfig()
    print(path_repo.to_dict())


