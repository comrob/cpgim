class Process:
    DERIVATIVE_PREFIX = "d_"

    def __init__(self, function_name, input_names, output_names):
        # retyping
        if isinstance(output_names, str):
            output_names = (output_names, )
        if isinstance(input_names, str):
            input_names = (input_names, )

        assert isinstance(input_names, tuple), "input_names must be tuple"
        assert isinstance(output_names, tuple), "output_names must be tuple"
        assert len(output_names) > 0, "Process must output something."
        assert all([self.is_variable_name_derivative(outn) == self.is_variable_name_derivative(output_names[0])
                    for outn in output_names]), "Process must represent either differential equation or an expression."
        assert all([not self.is_variable_name_derivative(inn) for inn in input_names]), \
            "Some of the input names {} contain derivative prefix {}.".format(input_names, self.DERIVATIVE_PREFIX)
        assert len(set(output_names)) + len(set(input_names)) == len(set(list(output_names) + list(input_names))), \
            "Recursion found in inp:{} outp:{}".format(input_names, output_names)

        self._differential = self.is_variable_name_derivative(output_names[0])  # FIXME: does not consider multiple outp


        self.function_name = function_name
        self.input_names = input_names
        self.output_names = output_names

        self._tupleing_function = True
        self._caller = self._call_first_time

        assert len(set(self.input_names + self.output_names)) == len(set(self.input_names) | set(self.output_names)), \
            "Recursive variable detected. See input:{} output:{}".format(input_names, output_names)

        self._function = None

    def set_function(self, functions):
        assert isinstance(functions, dict)
        if self.function_name in functions:
            if callable(functions[self.function_name]):
                self._function = functions[self.function_name]
            else:
                raise ValueError("Casted function {} should be callable.".format(self.function_name))
        else:
            raise KeyError("No function of name {} found in function list: {}".format(
                self.function_name, functions.keys()))
        return self

    def set_function_tupleing(self, is_tupleying):
        """
        Lots functions return just one thing. It is pain to rewrite the function to return tuple that contains that
        one thing. Thus it is better to just adapt the Process that calls this
        function and returns its value as a tuple.
        :param is_tupleying:
        :return:
        """
        if is_tupleying:
            self._tupleing_function = True
            self._caller = self._call_tupleing_function
        else:
            self._tupleing_function = False
            self._caller = self._call_non_tupleing_function
        return self

    def to_tuple(self):
        self.set_function_tupleing(False)
        return self

    def _get_input_values(self, state):
        return tuple([state[vn] for vn in self.input_names])

    def is_depending_on(self, process):
        """
        True if self is dependent on 'process'.
        :param process:
        :return:
        """
        assert isinstance(process, Process)
        for outp in process.output_names:
            if outp in self.input_names:
                return True
        return False

    def is_differential(self):
        return self._differential

    @staticmethod
    def dependence_ordering(processes):
        """
        Sorts the list of processes in such a way that each process is not dependent on any process on the right.
        It is assumed that there are no cyclic dependencies.
        :param processes: array of processes
        :return: ordered list of processes (does not modify the original list!)
        """
        n = len(processes)
        indexes = [i for i in range(n)]
        ordering = [-1] * n
        is_first = [True] * n
        depedency_table = [[False] * n for i in range(n)]
        for i in indexes:
            for j in indexes:
                if processes[i].is_depending_on(processes[j]):
                    is_first[j] = False
                    depedency_table[i][j] = True

        open_set = [indx for indx in indexes if is_first[indx]]
        for op in range(n * n):
            if op == len(open_set):
                break
            ordering[open_set[op]] = op
            for i in indexes:
                if depedency_table[open_set[op]][i]:
                    open_set.append(i)

        srt = sorted(zip(indexes, ordering), key=lambda p: p[1], reverse=True)
        return [processes[p[0]] for p in srt]

    @classmethod
    def is_variable_name_derivative(cls, variable_name):
        return cls.DERIVATIVE_PREFIX in variable_name[0:len(cls.DERIVATIVE_PREFIX)]

    @classmethod
    def variable_name_integrated(cls, variable_name):
        if cls.is_variable_name_derivative(variable_name):
            return variable_name[len(cls.DERIVATIVE_PREFIX):]
        else:
            return variable_name

    def __str__(self):
        return "{}({})->{}".format(self.function_name,
                                   ", ".join(self.input_names),
                                   ", ".join(self.output_names))

    def _call_tupleing_function(self, state, **kwargs):
        return self._function(*self._get_input_values(state))

    def _call_non_tupleing_function(self, state, **kwargs):
        # for function that doesn't return tuple
        return self._function(*self._get_input_values(state)),

    def _call_first_time(self, state, **kwargs):
        """
        Runtime check whether the function returns tuples or not, if not, the automatic tupleying is activated.
        :param state:
        :param kwargs:
        :return:
        """
        # for function that we don't know whether it returns tuple or not
        try:
            ret = self._function(*self._get_input_values(state))
            is_tupleing = type(ret) == tuple
            self.set_function_tupleing(is_tupleing)
            if is_tupleing:
                return ret
            else:
                return ret,
        except TypeError as err:
            print("Process {} failed at first run: {}".format(self, err))
            exit(1)

    def __call__(self, state, **kwargs):
        return self._caller(state, **kwargs)

    def __eq__(self, other):
        if type(other) == self.__class__:
            return all([self.function_name == other.function_name,
                        self.output_names == other.output_names,
                        self.input_names == other.input_names])
        else:
            return False
