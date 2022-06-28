from typing import Union, Tuple, Optional, Dict, List
from dynsys_framework.dynamic_system import process
IONames = Union[str, Tuple[str, ...]]
FunctionRef = Union[callable, str]
FunctionDict = Dict[str, callable]
InitDict = Dict[str, object]


class ModelExecutorParameters:
    """
    Parametrization class which is able to generate containers that are inputs for the Model and ModelExecutor robot_utils.
    After parametrization the user calls generate functions which create complete containers.
    """
    def __init__(self):
        self._parameters = []
        self._common_functions = {}

    def add(self, output_names: IONames, input_names: IONames, function: FunctionRef,
            default_initial_values: Optional[InitDict] = None):
        self._parameters.append(Parameter(output_names, input_names, function,
                                          default_initial_values=default_initial_values))

    def set_common_functions(self, common_functions: FunctionDict):
        self._common_functions = common_functions

    def _extract_anonymous_functions(self) -> FunctionDict:
        """
        Extracts anonymous functions from the parameters into a dictionary.
        Raises error if the function names clashes (or are not present in case of common functions)
        :return: Dictionary of anonymous functions.
        """
        anonymous_functions = {}
        for parameter in self._parameters:
            function_name = parameter.function_name
            function_lambda = parameter.function_lambda
            uses_common_function = parameter.uses_common_function
            if uses_common_function:
                if function_name not in self._common_functions.keys():
                    raise ValueError(
                        "{} flagged as common function is not defined in the common function dictionary".format(
                            function_name, self._common_functions.keys()
                        ))
            else:
                if function_name in self._common_functions.keys():
                    raise ValueError("anonymous function {} clashes with name in the common_functions {}".format(
                        function_name, self._common_functions.keys()
                    ))
                if function_name in anonymous_functions.keys():
                    raise ValueError("anonymous function {} clashes with name in the anonymous_functions {}".format(
                        function_name, anonymous_functions.keys()
                    ))
                anonymous_functions[function_name] = function_lambda
        return anonymous_functions

    def generate_function_dictionary(self) -> FunctionDict:
        anonymous_functions = self._extract_anonymous_functions()
        return {**anonymous_functions, **self._common_functions}

    def generate_processes(self) -> List[process.Process]:
        return [param.extract_process() for param in self._parameters]

    def generate_initializations(self) -> InitDict:
        initializations = {}
        for param in self._parameters:
            for k in param.default_initial_values:
                if k in initializations.keys():
                    raise ValueError("the variable {} initialization is already defined in {}".format(
                        k, initializations.keys()
                    ))
                initializations[k] = param.default_initial_values[k]
        return initializations


class Parameter:
    def __init__(self, output_names: IONames, input_names: IONames, function: FunctionRef,
                 default_initial_values: Optional[InitDict] = None):
        self.function_lambda = None
        if type(function) == str:
            self.uses_common_function = True
            self.function_name = function
        else:
            self.uses_common_function = False
            self.function_name = self.anonymous_function_name_from_output(output_names)
            self.function_lambda = function

        self.input_names = input_names
        self.output_names = output_names
        if type(input_names) is str:
            self._input_names = (input_names, )
        else:
            self._input_names = input_names
        if type(output_names) is str:
            self._output_names = (output_names, )
        else:
            self._output_names = output_names

        if default_initial_values is not None:
            for k in default_initial_values.keys():
                assert process.Process.DERIVATIVE_PREFIX + k in self._output_names, \
                    "default initialization can be defined only for integrated variables." \
                    " {} has no gradient here.".format(k)
            self.default_initial_values = default_initial_values
        else:
            self.default_initial_values = {}

        for out_name in self._output_names:
            if process.Process.is_variable_name_derivative(out_name):
                assert process.Process.variable_name_integrated(out_name) in self.default_initial_values.keys(), \
                    "The differential variable {} has not initialization in {}.".format(
                        out_name, self.default_initial_values.keys()
                    )

    def extract_process(self) -> process.Process:
        return process.Process(self.function_name, self.input_names, self.output_names)

    @staticmethod
    def anonymous_function_name_from_output(output_names: IONames) -> str:
        if type(output_names) is str:
            return "<{}>".format(output_names)
        elif type(output_names) is tuple:
            return "<{}>".format(",".join(output_names))
        else:
            raise NotImplementedError("Not implemented")
