import unittest
from dynsys_framework.dynamic_system.model_executor import Model, ModelExecutor
from dynsys_framework.dynamic_system.process import Process
from dynsys_framework.dynamic_system.solvers import euler_solver


class TestModelExecutor(unittest.TestCase):

    @staticmethod
    def get_functions():
        functions = {
            "cons": lambda x: (1,),
            "stab": lambda x: (-x,),
            "unstab": lambda x: (x,),
            "lin_osc": lambda x, y: (y, -x),
        }
        return functions

    def test_model_executor_simple(self):
        functions = self.get_functions()
        process1 = Process("cons", ("a_1",), ("d_a_1",))
        process2 = Process("cons", ("a_2",), ("d_a_2",))
        process3 = Process("cons", ("a_2",), ("d_a_3",))
        model = Model(functions, [process1, process2, process3])

        model_executor = ModelExecutor(model, euler_solver(1), {})

        self.assertEqual(0, len(model_executor._external_variable_names))
        self.assertEqual(6, len(
            set(model_executor._available_variable_names + ["a_1", "a_2", "a_3", "d_a_1", "d_a_2", "d_a_3"])))
        self.assertEqual(2, len(set(model_executor._all_model_inputs + ["a_1", "a_2"])))

        state = {"a_1": 0, "a_2": 1, "a_3": 2}
        model_executor.initialize_integrating_variables(state)

        model_executor()

        new_state = model_executor.get__next_integrated_variable_values()
        expected_new_state = {"a_1": 1, "a_2": 2, "a_3": 3}
        self.assertDictEqual(expected_new_state, new_state)

        model_executor()

        new_state = model_executor.get__next_integrated_variable_values()
        expected_new_state = {"a_1": 2, "a_2": 3, "a_3": 4}
        self.assertDictEqual(expected_new_state, new_state)

        model_executor()
        model_executor()
        model_executor()

        new_state = model_executor.get__next_integrated_variable_values()
        expected_new_state = {"a_1": 5, "a_2": 6, "a_3": 7}
        self.assertDictEqual(expected_new_state, new_state)

    def test_err_duplicate_available_variables(self):
        functions = self.get_functions()
        process1 = Process("cons", ("a_1",), ("d_a_1",))
        model = Model(functions, [process1])
        ModelExecutor(model, euler_solver(1), {})
        ModelExecutor(model, euler_solver(1), {"a_2": lambda: 2})
        with self.assertRaises(AssertionError):
            ModelExecutor(model, euler_solver(1), {"a_1": lambda: 2})

    def test_err_required_not_available(self):
        functions = self.get_functions()
        process1 = Process("cons", ("a_2",), ("a_1",))
        model = Model(functions, [process1])
        ModelExecutor(model, euler_solver(1), {"a_2": lambda: 2})
        with self.assertRaises(AssertionError):
            ModelExecutor(model, euler_solver(1), {})

    def test_external_binding(self):
        functions = self.get_functions()
        process1 = Process("stab", ("a_1",), ("d_a_2",))
        model_exec = ModelExecutor(Model(functions, [process1]), euler_solver(1), {"a_1": lambda: 2})
        model_exec.initialize_integrating_variables({"a_2": 1})
        model_exec()
        self.assertEqual(-1, model_exec.get__next_integrated_variable_values()["a_2"])
        self.assertEqual(2, model_exec.get_external_variable_values()["a_1"])
        model_exec()
        self.assertEqual(-3, model_exec.get__next_integrated_variable_values()["a_2"])
        self.assertEqual(2, model_exec.get_external_variable_values()["a_1"])

    def test_two_bound_models_executor(self):
        functions = self.get_functions()
        process1 = Process("lin_osc", ("a_1", "a_2"), ("d_a_1", "d_a_2"))
        process2 = Process("unstab", ("a_2",), ("d_a_1",))
        process3 = Process("stab", ("a_1",), ("d_a_2",))

        model_exec1 = ModelExecutor(Model(functions, [process1]), euler_solver(1), {})

        glob_state = {"x": None, "y": None}

        # model2 controls "a_1" but needs "a_2"
        model_exec2 = ModelExecutor(Model(functions, [process2]), euler_solver(1), {"a_2": lambda: glob_state["x"]})
        # model3 controls "a_2" but needs "a_1"
        model_exec3 = ModelExecutor(Model(functions, [process3]), euler_solver(1), {"a_1": lambda: glob_state["y"]})

        state = {"a_1": 1, "a_2": 2}
        model_exec1.initialize_integrating_variables(state)
        model_exec2.initialize_integrating_variables(state)
        model_exec3.initialize_integrating_variables(state)

        for i in range(10):
            model_exec1()
            glob_state["x"] = model_exec3.get__next_integrated_variable_values()["a_2"]
            glob_state["y"] = model_exec2.get__next_integrated_variable_values()["a_1"]
            model_exec2()
            model_exec3()

            self.assertEqual(model_exec1.get__next_integrated_variable_values()["a_2"],
                             model_exec3.get__next_integrated_variable_values()["a_2"])
            self.assertEqual(model_exec1.get__next_integrated_variable_values()["a_1"],
                             model_exec2.get__next_integrated_variable_values()["a_1"])
