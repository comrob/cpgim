from dynsys_framework.dynamic_system.process import Process
from dynsys_framework.dynamic_system.expression_tree import ExpressionTree


class Model:
    def __init__(self, functions, processes: [Process]):
        """
        :param functions: dictionary string -> fun(vars)->dvars
        :param process_declarations:
        """
        for p in processes:
            assert isinstance(p, Process), "all processes should be of class Process. {} is not Process".format(p)
        assert isinstance(functions, dict), "functions is a dictionary fun_name->callable"
        # assert all([callable(f) for f in functions.values()]), "all functions values must be callable"
        for f in functions:
            assert callable(functions[f]), "function {} is not callable".format(f)

        # assert all(p.function_name in functions.keys() for p in processes), "required function not found in functions"
        for p in processes:
            assert p.function_name in functions.keys(), "function {} is not in {}".format(
                p.function_name, functions.keys())

        self._processes = processes
        self._differential_processes = []
        self._expression_processes = []
        for pi in processes:
            if pi.is_differential():
                self._differential_processes.append(pi)
            else:
                self._expression_processes.append(pi)

        self._expression_tree = ExpressionTree(self._expression_processes)

        self._diff_output_variable_names = [name
                                            for process in self._differential_processes
                                            for name in process.output_names]

        self._intg_output_variable_names = [Process.variable_name_integrated(name)
                                            for name in self._diff_output_variable_names]

        self._diff_to_intg = dict(zip(self._diff_output_variable_names, self._intg_output_variable_names))

        self._expr_output_variable_names = [name
                                            for process in self._expression_processes
                                            for name in process.output_names]

        self._clean_output_names = self._diff_output_variable_names + self._expr_output_variable_names

        self._output_names = self._clean_output_names + self._intg_output_variable_names

        assert len(self._output_names) == len(set(self._output_names)), \
            "there is a duplicate in output_names: {}".format(self._output_names)
        self._output_names = set(self._output_names)

        self._diff_input_variable_names = [name
                                           for process in self._differential_processes
                                           for name in process.input_names]

        self._expr_input_variable_names = [name
                                           for process in self._expression_processes
                                           for name in process.input_names]

        self._input_names = set(self._diff_input_variable_names + self._expr_input_variable_names)

        self._required_state_variables = set(self.required_variable_binding() + self.required_variable_initialization())

        self.functions = functions

        for pd in self._processes:
            pd.set_function(self.functions)

    def input_names(self):
        """
        Input names for all processes.
        :return:
        """
        return self._input_names

    def output_names(self):
        """
        Output names of all processes and with integrated derivative variables.
        :return:
        """
        return self._output_names

    def expression_output_names(self):
        return self._expr_output_variable_names

    def derivative_output_names(self):
        return self._diff_output_variable_names

    def integrated_output_names(self):
        return self._intg_output_variable_names

    def clean_output_names(self):
        """
        Output names of all processes.
        :return:
        """
        return self._clean_output_names

    def required_variable_binding(self):
        """
        Variable names that must be given from external source.
        :return:
        """
        return [name for name in self._input_names if not name in self._output_names]

    def required_variable_initialization(self):
        """
        Variable names that must be initialised.
        :return:
        """
        return self._intg_output_variable_names

    def get_derivative_to_integrated_dictionary(self):
        """
        Dictionary derivated_variable_name -> (integrated)variable_name.
        :return:
        """
        return self._diff_to_intg

    def __str__(self):
        ret =  "------------------------FUNCTIONS({}):\n".format(len(self.functions))
        ret += "; \n".join(["{}:= {}".format(k, self.functions[k]) for k in self.functions])
        ret += "\n------------------------PROCESSES({}):\n".format(len(self._processes))
        ret += "EXPRESSIONS:\n"
        ret += "; \n".join([str(proc) for proc in self._expression_tree._processes])
        ret += "\nDERIVATIVES:\n"
        ret += "; \n".join([str(proc) for proc in self._differential_processes])
        ret += "\n------------------------VARIABLES({}):".format(
            len(self._intg_output_variable_names) +
            len(self.required_variable_binding()) +
            len(self._diff_output_variable_names) +
            len(self.expression_output_names())
        )
        ret += "\nINTEGRATED:{}".format(self._intg_output_variable_names)
        ret += "\nEXTERNAL:{}".format(self.required_variable_binding())
        ret += "\nDERIVATIVES:{}".format(self._diff_output_variable_names)
        ret += "\nEXPRESSIONS:{}".format(self.expression_output_names())
        return ret

    def _eval_differential_processes(self, state):
        names = []
        values = []

        for pi in self._differential_processes:
            values += pi(state)
            names += pi.output_names
        return dict(zip(names, values))

    def __call__(self, state, **kwargs):
        _state = dict((k, state[k]) for k in state)
        _state = self._expression_tree(_state)
        _state = {**_state, **self._eval_differential_processes(_state)}
        return _state

