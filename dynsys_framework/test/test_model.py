import unittest
from dynsys_framework.dynamic_system.model import Process, Model


class TestModel(unittest.TestCase):

    @staticmethod
    def get_functions():
        functions = {
            "sum": lambda x, y: (x + y,),
            "cons": lambda x: (1,),
            "stab": lambda x: (-x,),
            "unstab": lambda x: (x,),
            "lin_osc": lambda x, y: (y, -x),
        }
        return functions

    def test_model_simple(self):
        functions = self.get_functions()
        process1 = Process("cons", ("a_1",), ("d_a_1",))
        process2 = Process("cons", ("a_2",), ("d_a_2",))
        process3 = Process("cons", ("a_2",), ("d_a_3",))
        model = Model(functions, [process1, process2, process3])

        self.assertEqual(2, len(set(model.input_names() | {"a_1", "a_2"})))
        self.assertEqual(6, len(set(model.output_names() | {"d_a_1", "d_a_2", "d_a_3", "a_1", "a_2", "a_3"})))

        state = {"a_1": 0, "a_2": 1, "a_3": 2, "a_20": 2}

        dstate = model(state)

        self.assertIsInstance(dstate, dict)
        self.assertEqual(1, dstate["d_a_1"])
        self.assertEqual(1, dstate["d_a_2"])
        self.assertEqual(1, dstate["d_a_3"])
        self.assertEqual(0, dstate["a_1"])
        self.assertEqual(1, dstate["a_2"])
        self.assertEqual(2, dstate["a_3"])
        self.assertEqual(2, dstate["a_20"])

    def test_err_model_duplicate_output(self):
        functions = self.get_functions()
        process1 = Process("cons", ("a_1",), ("b_1",))
        process2 = Process("cons", ("a_2",), ("b_1",))
        with self.assertRaises(AssertionError):
            Model(functions, [process1, process2])

    def test_err_model_noncallable_function(self):
        functions = self.get_functions()
        process1 = Process("cons", ("a_1",), ("b_1",))
        with self.assertRaises(AssertionError):
            Model({**functions, **{"none": None}}, [process1])

    def test_model_equivalent_eqs(self):
        functions = self.get_functions()
        process1 = Process("stab", ("a_1",), ("d_a_2",))
        process2 = Process("unstab", ("a_2",), ("d_a_1",))
        process3 = Process("lin_osc", ("a_1", "a_2"), ("d_a_1", "d_a_2"))
        model1 = Model(functions, [process1, process2])
        model2 = Model(functions, [process3])
        state = {"a_1": 1, "a_2": 2}
        dstate1 = model1(state)
        dstate2 = model2(state)
        expected = {"d_a_1": 2, "d_a_2": -1}
        self.assertEqual(expected["d_a_1"], dstate1["d_a_1"])
        self.assertEqual(expected["d_a_1"], dstate2["d_a_1"])
        self.assertEqual(expected["d_a_2"], dstate1["d_a_2"])
        self.assertEqual(expected["d_a_2"], dstate2["d_a_2"])

    def test_hybrid_model(self):
        functions = self.get_functions()
        process1 = Process("stab", ("b_1",), ("d_a_2",))
        process2 = Process("unstab", ("a_2",), ("d_a_1",))
        process3 = Process("sum", ("a_1", "c"), ("b_1",))

        model = Model(functions, [process1, process2, process3])

        self.assertEqual(4, len(set(model.input_names() | {"a_1", "a_2", "b_1", "c"})))
        self.assertEqual(5, len(set(model.output_names() | {"d_a_1", "d_a_2", "a_1", "a_2", "b_1"})))

        self.assertEqual(2, len(set(model.required_variable_initialization() + ["a_1", "a_2"])))
        self.assertEqual(1, len(set(model.required_variable_binding() + ["c"])))

        state = {"a_1": 1, "a_2": 2, "c": 3}
        new_state = model(state)
        expected = {"a_1": 1, "a_2": 2, "c": 3, "d_a_1": 2, "b_1": 4, "d_a_2": -4}
        for k in expected:
            self.assertEqual(expected[k], new_state[k], "{}:{} =/= {}:{}".format(k, expected[k], k, new_state[k]))

    def test_same_integrated_and_expression_names_err(self):
        process1 = Process("sum", ("a_1", "c"), ("b_1",))
        process2 = Process("sum", ("a_1", "c"), ("d_b_1",))
        with self.assertRaises(AssertionError):
            Model(self.get_functions(), [process1, process2])
