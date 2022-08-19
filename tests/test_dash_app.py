import unittest
from unittest import TestCase
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

from app import convert_to_numpy_valdata, generate_N_grams


class Test_dash_app(TestCase):

    def test_convert_to_numpy_valdata(self):
        data_input = {'We can develop a better system': 0}
        data_output = convert_to_numpy_valdata(data_input)
        np.testing.assert_equal(data_output[0], 'We can develop a better system')
        np.testing.assert_equal(data_output[1], 0)

    def test_generate_n_grams(self):
        data_input = (
            np.array(['We can develop a better system']),
            np.array([0])
        )
        data_output = generate_N_grams(data_input)
        data_op_expected = [[['*', '*', 'We', 'can', 'develop']],
                            [['*', 'We', 'can', 'develop', 'a']],
                            [['We', 'can', 'develop', 'a', 'better']],
                            [['can', 'develop', 'a', 'better', 'system']],
                            [['develop', 'a', 'better', 'system', '*']],
                            [['a', 'better', 'system', '*', '*']]]

        self.assertEqual(data_output[0], data_op_expected)
        self.assertEqual(data_output[1], [0, 0, 0, 0, 0, 0])
        pass


