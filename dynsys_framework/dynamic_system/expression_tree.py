from dynsys_framework.dynamic_system.process import Process


class ExpressionTree:
    def __init__(self, processes):
        """
        :param processes:
        """
        assert all([isinstance(p, Process) for p in processes]), "all processes should be of class Process"

        self._output_names = []
        self._all_input_names = []
        for pi in processes:
            self._output_names += pi.output_names
            self._all_input_names += pi.input_names
        assert len(set(self._output_names)) == len(self._output_names), \
            "there are duplicates in outputs {}".format(self._output_names)

        self._input_names = [inp for inp in self._all_input_names if not inp in self._output_names]

        self._processes = Process.dependence_ordering(processes)

        for i in range(len(self._processes)):
            for j in range(i + 1, len(self._processes)):
                assert not self._processes[i].is_depending_on(self._processes[j]), \
                    "Possible cyclic dependency in expression processes {} -> {}\n process order [{}]".format(
                        self._processes[i], self._processes[j], "\n".join([str(_p) for _p in self._processes]))

    def set_functions(self, functions):
        for process in self._processes:
            process.set_function(functions)

    def get_input_names(self):
        """
        Input variable names required by this expression tree.
        :return:
        """
        return self._input_names

    def get_output_names(self):
        """
        All output variable names provided by this expression tree.
        :return:
        """
        return self._output_names

    def __call__(self, state, *args, **kwargs):
        for pi in self._processes:
            values = pi(state)
            names = pi.output_names
            for nv in zip(names, values):
                state[nv[0]] = nv[1]
        return state
