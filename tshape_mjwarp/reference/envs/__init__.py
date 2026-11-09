"""Easy cooperative transport."""


def parallel_env(**kwargs):
    from .tshape_env import TShapeParallelEnv
    return TShapeParallelEnv(**kwargs)


__all__ = ["parallel_env"]
