from dynsys_framework.dynamic_system.process import Process
import unittest
import random
import itertools


class TestProcess(unittest.TestCase):

    def test_dependency(self):
        p1 = Process("f", ("a", "b"), ("c", ))
        p2 = Process("f", ("q","d"), ("a", ))
        p3 = Process("f", (), ("b", ))
        self.assertTrue(p1.is_depending_on(p2))
        self.assertFalse(p2.is_depending_on(p1))
        self.assertFalse(p3.is_depending_on(p1))
        self.assertFalse(p3.is_depending_on(p2))

    def test_dependency_small_ordering(self):
        p1 = Process("f", ("a", "b", "c"), ("d",))
        p2 = Process("f", ("b", "c"), ("a",))
        expected = [p2, p1]
        processes = [p1, p2]
        processes = Process.dependence_ordering(processes)
        self.assertListEqual(expected, processes)

    def test_dependency_small_ordering2(self):
        p1 = Process("f", ("x",), ("b",))
        p2 = Process("f", ("c",), ("a",))
        p3 = Process("f", ("b",), ("c",))
        processes = [p1, p2, p3]
        processes = Process.dependence_ordering(processes)
        for i in range(len(processes)):
            for j in range(i + 1, len(processes)):
                self.assertFalse(processes[i].is_depending_on(processes[j]),
                                 "Wrong dependency {} => {}.\n Before {} \n After {}".format(
                                     processes[i], processes[j], [str(pi) for pi in [p1, p2, p3]],
                                     [str(pi) for pi in processes]
                                 ))

    def test_dependency_ordering(self):
        p1 = Process("f", ("a", "b", "c"), ("d",))
        p2 = Process("f", ("b", "c"), ("a",))
        p3 = Process("f", ("c",), ("b",))
        p4 = Process("f", ("q",), ("c",))
        p5 = Process("f", ("m",), ("n",))
        processes = [p1, p5, p2, p4, p3]
        random.shuffle(processes)
        processes = Process.dependence_ordering(processes)
        for i in range(len(processes)):
            for j in range(i + 1, len(processes)):
                self.assertFalse(processes[i].is_depending_on(processes[j]))

    def test_complex_dependency_ordering(self):
        processes = [
            Process("mlt", ("z1", "x4"), ("z3",)),
            Process("sum", ("z3", "a1"), ("b1",)),
            Process("num", (), ("a1", "a2", "x3", "x4")),
            Process("sum", ("x1", "x2"), ("y",)),
            Process("mlt", ("z2", "x3"), ("z4",)),
            Process("sum", ("b2", "c"), ("d",)),
            Process("sum", ("z4", "a2"), ("b2",)),
            Process("div", ("x3", "x4"), ("z1", "z2")),
        ]
        for x in range(20):
            random.shuffle(processes)
            orig_processes = [pi for pi in processes]
            processes = Process.dependence_ordering(processes)
            for i in range(len(processes)):
                for j in range(i + 1, len(processes)):
                    self.assertFalse(processes[i].is_depending_on(processes[j]), "Wrong dependency {} => {}.\n Before {} \n After {}".format(
                        processes[i], processes[j], [str(pi) for pi in orig_processes], [str(pi) for pi in processes]
                    ))

    def test_differential_process(self):
        p1 = Process("f", ("x",), ("d_b",))
        p2 = Process("f", ("x",), ("b",))

        self.assertTrue(p1.is_differential())
        self.assertFalse(p2.is_differential())
        self.assertEqual("b_1_wdw_45w", Process.variable_name_integrated("d_b_1_wdw_45w"))
        self.assertEqual("b_1_wdw_45w", Process.variable_name_integrated("b_1_wdw_45w"))

    def test_hybrid_process_error(self):
        with self.assertRaises(AssertionError):
            Process("f", ("x",), ("y", "d_b"))

    def test_derivative_input_error(self):
        with self.assertRaises(AssertionError):
            Process("f", ("d_x",), ("y", ))

    def test_recursion_error(self):
        with self.assertRaises(AssertionError):
            Process("f", ("a", "b", "c"), ("y", "z", "a"))
