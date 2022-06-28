import numpy as np


class UpdatableProvider(object):
    def update(self):
        pass


class GenericUpdatableProvider(UpdatableProvider):
    def __init__(self, function: callable, init_argument=None):
        self._function = function
        self._argument = init_argument
        self._argument_getter = None
        self._first_argument_getter = None
        self._updater = self._first_update

    def set_argument_getter(self, argument_getter: callable, init_value=None):
        self._argument_getter = argument_getter
        if init_value is not None:
            self._first_argument_getter = lambda: init_value
        else:
            self._first_argument_getter = argument_getter

    def _first_update(self):
        self._argument = self._first_argument_getter()
        self._updater = self._update

    def _update(self):
        self._argument = self._argument_getter()

    def update(self):
        self._updater()

    def __call__(self):
        return self._function(self._argument)


class ProviderContainer(UpdatableProvider):
    def __init__(self):
        self._providers = []

    def add_provider(self, provider: UpdatableProvider):
        self._providers.append(provider)

    def update(self):
        for prov in self._providers:
            prov.update()

    def __call__(self) -> np.ndarray:
        return np.asarray([prov() for prov in self._providers])

    def __str__(self):
        return "ProviderContainer: " + ",".join(self._providers)
