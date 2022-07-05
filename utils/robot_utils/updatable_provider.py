import numpy as np
from typing import Callable

FEMURS = [2, 3, 6, 7, 10, 11]
COXAS = [0, 1, 4, 5, 8, 9]
LEFT = [0, 2, 5]
RIGHT = [1, 3, 4]

DynamicProvider = Callable[[], np.ndarray]


class UpdatableProvider:
    def set_data_getter(self, getter: callable):
        pass

    def update(self):
        pass


class ProviderContainer(UpdatableProvider):
    def __init__(self):
        self._providers = []
        self._named_providers = {}

    def add_provider(self, provider: UpdatableProvider, name: str):
        self._providers.append(provider)
        self._named_providers[name] = provider

    def get_provider(self, name) -> UpdatableProvider:
        return self._named_providers[name]

    def update(self):
        for prov in self._providers:
            prov.update()

    def __call__(self) -> np.ndarray:
        return np.asarray([prov() for prov in self._providers])

    def __str__(self):
        return "PerceptionArray: " + ",".join(self._providers)