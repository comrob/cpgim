import unittest
from dynsys_framework.execution_helpers.model_executor_parameters import Parameter, ModelExecutorParameters
from dynsys_framework.dynamic_system.process import Process


class TestModelExecutorParameters(unittest.TestCase):

    def test_parameter_simple(self):
        param1 = Parameter("fruits", ("apples", "pears"), lambda x, y: x + y)
        self.assertEqual(False, param1.uses_common_function)
        self.assertTrue(Process("<fruits>", ("apples", "pears"), "fruits") == param1.extract_process())
        self.assertDictEqual({}, param1.default_initial_values)

        param2 = Parameter(("d_fruits", "d_baskets"), ("apples", "pears"), lambda x, y: x + y,
                           default_initial_values={"fruits": 0, "baskets": 0})
        self.assertEqual(False, param2.uses_common_function)
        self.assertTrue(
            Process("<d_fruits,d_baskets>", ("apples", "pears"), ("d_fruits", "d_baskets")) == param2.extract_process()
        )
        self.assertDictEqual({"fruits": 0, "baskets": 0}, param2.default_initial_values)

        param3 = Parameter("apple_foo", "apples", "foo")
        self.assertEqual(True, param3.uses_common_function)
        self.assertTrue(
            Process("foo", "apples", "apple_foo") == param3.extract_process()
        )
        self.assertDictEqual({}, param3.default_initial_values)

    def test_parameter_assertions(self):
        # missing default initialization
        with self.assertRaises(AssertionError):
            Parameter(("d_fruits", "d_baskets"), ("apples", "pears"), lambda x, y: x + y)

        # only one is initialized
        with self.assertRaises(AssertionError):
            Parameter(("d_fruits", "d_baskets"), ("apples", "pears"), lambda x, y: x + y,
                      default_initial_values={"fruits": 0})

        # wrong variable init
        with self.assertRaises(AssertionError):
            Parameter("d_fruits", ("apples", "pears"), lambda x, y: x + y,
                      default_initial_values={"ww": 0})

        # init of non-differential
        with self.assertRaises(AssertionError):
            Parameter("fruits", ("apples", "pears"), lambda x, y: x + y,
                      default_initial_values={"fruits": 0})

        # default initialization of unknown variable
        with self.assertRaises(AssertionError):
            Parameter("fruits", ("apples", "pears"), lambda x, y: x + y, default_initial_values={"ww": 0})

    def test_model_executor_parameters_simple(self):
        mep = ModelExecutorParameters()
        mep.add("fruits", ("apples", "pears"), lambda x, y: x + y)
        mep.add(("d_fruits", "d_baskets"), ("apples", "pears"), lambda a, b: (a*b, a+b),
                default_initial_values={"fruits": 0, "baskets": 0})
        mep.add("d_apple_foo", "apples", "foo", default_initial_values={"apple_foo": 1})

        mep.set_common_functions({"foo": lambda a: a})

        expected_fun_dictionary = {
            "foo": lambda a: a,
            "<fruits>": lambda x, y: x + y,
            "<d_fruits,d_baskets>": lambda a, b: (a*b, a+b)
        }

        self.assertSetEqual(set(expected_fun_dictionary.keys()), set(mep.generate_function_dictionary().keys()))

        expected_processes = [
            Process("<fruits>", ("apples", "pears"), "fruits"),
            Process("<d_fruits,d_baskets>", ("apples", "pears"), ("d_fruits", "d_baskets")),
            Process("foo", "apples", "d_apple_foo"),
        ]

        self.assertListEqual(expected_processes, mep.generate_processes())

        expected_initializations = {"fruits": 0, "baskets": 0, "apple_foo": 1}

        self.assertDictEqual(expected_initializations, mep.generate_initializations())

    def test_model_executor_parameters_assetions(self):
        mep = ModelExecutorParameters()
        mep.add("fruits", ("apples", "pears"), lambda x, y: x + y)
        mep.add(("d_fruits", "d_baskets"), ("apples", "pears"), lambda a, b: (a*b, a+b),
                default_initial_values={"fruits": 0, "baskets": 0})
        mep.add("d_apple_foo", "apples", "foo", default_initial_values={"apple_foo": 1})

        # missing common function
        with self.assertRaises(ValueError):
            mep.generate_function_dictionary()

        mep.set_common_functions({"foo": lambda a: a})
        mep.generate_function_dictionary()  # no exception

        # clash with anonymous name <fruits>
        mep.set_common_functions({"foo": lambda a: a, "<fruits>": lambda x, y: x + y})
        with self.assertRaises(ValueError):
            mep.generate_function_dictionary()

        mep.set_common_functions({"foo": lambda a: a})
        mep.generate_function_dictionary()  # no exception

        # another <fruits> anonymous function created by adding new Process with the fruits output
        mep.add( "fruits", ("oranges", "pears"), lambda w, v: v - w)
        with self.assertRaises(ValueError):
            mep.generate_function_dictionary()

        # same variable inited twice
        mep.add("d_apple_foo", "oranges", "foo", default_initial_values={"apple_foo": 1})
        with self.assertRaises(ValueError):
            mep.generate_initializations()




