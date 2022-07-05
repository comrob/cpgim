from dynsys_framework.dynamic_system.model import Model


class ModelExecutor:
    """
    Binds external input into the model and executes it when called.
    Each call executor collects external values, evaluates expressions and integrates differential equations.
    """

    def __init__(self, model: Model, solver, external_binders):
        assert isinstance(model, Model), "model must be instance of Model"
        assert isinstance(external_binders, dict), "external_binders must be a dict str->callable"
        assert all([callable(b) for b in external_binders.values()]), "external_binders values must be callable"
        assert callable(solver), "solver must be callable"

        self._model = model
        self._solver = solver
        self._external_binders = external_binders

        self._all_model_inputs = list(model.input_names())

        # collecting names of all variables that will be updated

        self._external_variable_names = list(external_binders.keys())
        self._expression_names = model.expression_output_names()
        self._derivative_variable_names = model.derivative_output_names()
        self._integrated_variable_names = model.integrated_output_names()

        self._available_variable_names = self._external_variable_names + self._expression_names + \
                                         self._derivative_variable_names + self._integrated_variable_names

        # preparing storage for variable values - these four defines state at time 't'

        self._external_variable_values = dict((name, None) for name in self._external_variable_names)
        self._integrated_variable_values = dict((name, None) for name in self._integrated_variable_names)
        self._expression_values = dict((name, None) for name in self._expression_names)
        self._derivative_variable_values = dict((name, None) for name in self._derivative_variable_names)

        # this dictionary will store new integrated variable values computed by solver from state at time 't'
        self._next_integrated_variable_values = dict((name, None) for name in self._integrated_variable_names)

        assert len(self._available_variable_names) == len(set(self._available_variable_names)), \
            "there are duplicates in available_variables:" + str(self._available_variable_names)

        for required_var in self._all_model_inputs:
            assert required_var in self._available_variable_names, "required variable {} is not available in {}" \
                .format(required_var, self._available_variable_names)

    def initialize_integrating_variables(self, init_state):
        for k in self._integrated_variable_values:
            if k in init_state:
                self._next_integrated_variable_values[k] = init_state[k]
            else:
                raise KeyError("Missing variable name {} in init_state {}".format(k, init_state))

    def _set_internal_variable_values(self, internal_variable_values):
        for k in self._integrated_variable_values:
            self._integrated_variable_values[k] = self._next_integrated_variable_values[k]
            self._next_integrated_variable_values[k] = internal_variable_values[k]
        for k in self._expression_values:  # needed only for log purposes
            self._expression_values[k] = internal_variable_values[k]
        for k in self._derivative_variable_values:  # needed only for log purposes
            self._derivative_variable_values[k] = internal_variable_values[k]
        # _external_variable_values are updated in call

    def _update_external_variable_values(self):
        # TODO: policies for "waiting"
        for k in self._external_variable_values:
            self._external_variable_values[k] = self._external_binders[k]()

    def get_external_variable_values(self):
        return self._external_variable_values

    def get__next_integrated_variable_values(self):
        return self._next_integrated_variable_values

    def get_last_state(self):
        return {
            **self._integrated_variable_values,
            **self._external_variable_values,
            **self._expression_values,
            **self._derivative_variable_values
        }

    def __str__(self):
        ret = "\n------------------------EXTERNAL BINDING({}):\n".format(len(self._external_binders))
        ret += ";\n".join(["{}:= {}".format(k, self._external_binders[k]) for k in self._external_binders])
        ret += "\n"
        ret += str(self._model)
        return ret

    def __call__(self, *args, **kwargs):
        self._update_external_variable_values()
        state = {**self._next_integrated_variable_values, **self._external_variable_values}
        self._set_internal_variable_values(
            self._solver(self._model, state, self._model.get_derivative_to_integrated_dictionary()))
