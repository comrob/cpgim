from dynsys_framework.dynamic_system.expression_tree import ExpressionTree, Process
import unittest
import random


class TestProcess(unittest.TestCase):

    def test_simple(self):
        functions = {
            "sum": lambda a, b: (a + b,)
        }
        et = ExpressionTree([Process("sum", ("x", "y"), ("z",))])
        et.set_functions(functions)
        new_state = et({"x": 2, "y": 3, "z": None})
        expected = {"x": 2, "y": 3, "z": 5}
        for k in ["x", "y", "z"]:
            self.assertEqual(expected[k], new_state[k])

    def test_simple_cyclic_dependency_error(self):
        with self.assertRaises(AssertionError):
            et = ExpressionTree([
                Process("f", ("x", "y"), ("z",)),
                Process("f", ("x", "z"), ("x",)),
            ])

    def test_complex_expression(self):
        functions = {
            "sum": lambda a, b: (a + b,),
            "mlt": lambda a, b: (a * b,),
            "div": lambda a, b: (a / b, b / a),
            "num": lambda: (1, 2, 3, 4)
        }
        processes = [
            Process("sum", ("x1", "x2"), ("y",)),
            Process("div", ("x3", "x4"), ("z1", "z2")),
            Process("mlt", ("z1", "x4"), ("z3",)),
            Process("mlt", ("z2", "x3"), ("z4",)),
            Process("sum", ("z3", "a1"), ("b1",)),
            Process("sum", ("z4", "a2"), ("b2",)),
            Process("sum", ("b2", "c"), ("d",)),
            Process("num", (), ("a1", "a2", "x3", "x4"))
        ]

        random.shuffle(processes)

        et = ExpressionTree(processes)
        et.set_functions(functions)

        inp_names = set(et.get_input_names())
        exp_inp_names = {"x1", "x2", "c"}
        out_names = set(et.get_output_names())
        exp_out_names = {"y", "z1", "z2", "z3", "z4", "b1", "b2", "a1", "a2", "x3", "x4", "d"}

        self.assertEqual(exp_inp_names, inp_names)
        self.assertEqual(exp_out_names, out_names)

        in_state = {"x1": 5, "x2": 6, "c": 7}

        exp_out_state = {
            "y": 11,
            "z1": 3 / 4,
            "z2": 4 / 3,
            "z3": 3,
            "z4": 4,
            "b1": 4,
            "b2": 6,
            "a1": 1,
            "a2": 2,
            "x3": 3,
            "x4": 4,
            "d": 13
        }
        exp_out_state = {**exp_out_state, **in_state}
        out_state = et(in_state)
        self.assertEqual(exp_out_state, out_state)
        out_state = et(in_state)
        self.assertEqual(exp_out_state, out_state, "ExpressionTree is not stateless.")

    def test_complex_cycle_error(self):
        processes = [
            Process("sum", ("x1", "x2"), ("y",)),
            Process("div", ("x3", "x4"), ("z1", "z2")),
            Process("mlt", ("z1", "x4"), ("z3",)),
            Process("mlt", ("z2", "d"), ("z4",)),
            Process("sum", ("z3", "a1"), ("b1",)),
            Process("sum", ("z4", "a2"), ("b2",)),
            Process("sum", ("b2", "c"), ("d",)),
            Process("num", (), ("a1", "a2", "x3", "x4"))
        ]

        with self.assertRaises(AssertionError):
            et = ExpressionTree(processes)

    def test_duplicates_in_output_error(self):
        processes = [
            Process("sum", ("x1", "x2"), ("y",)),
            Process("div", ("x3", "x4"), ("z1", "z2")),
            Process("mlt", ("z1", "x4"), ("a1",)),
            Process("mlt", ("z2", "d"), ("z4",)),
            Process("sum", ("z3", "a1"), ("b1",)),
            Process("sum", ("z4", "a2"), ("b2",)),
            Process("sum", ("b2", "c"), ("d",)),
            Process("num", (), ("a1", "a2", "x3", "x4"))
        ]
        with self.assertRaises(AssertionError):
            et = ExpressionTree(processes)


