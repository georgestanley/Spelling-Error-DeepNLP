import unittest
from unittest import TestCase

import pandas as pd
import pandas.testing as pd_testing
from torch.utils.data import TensorDataset, DataLoader, Dataset


from lstm_spell_classifier_w_context import remove_punctuation, cleanup_data, generate_N_grams, convert_to_pytorch_dataset


class Test_lstm_w_context(TestCase):

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation('Can, I be ! at office.'), 'Can I be  at office')

    def test_cleanup_data(self):
        pd_testing.assert_frame_equal(cleanup_data(pd.DataFrame({'Freiburg, will not fall!!'}, columns=['text'])),
                                      pd.DataFrame({'Freiburg will not fall'}, columns=['text']))

    def test_generate_N_grams(self):
        dataset_old = pd.DataFrame({'the economy will be in distress'}, columns=['text'])
        dataset_new = pd.DataFrame([
            [['the', 'economy', 'will', 'be', 'in'], 0],
            [['economy', 'will', 'be', 'in', 'distress'], 0]
        ], columns=['inputs', 'labels'])
        pd_testing.assert_frame_equal(generate_N_grams(dataset_old, 5), dataset_new)

    def test_convert_to_pytorch_dataset(self):
        data = pd.DataFrame([
            [['the', 'economy', 'will', 'be', 'in'], 0],
            [['economy', 'will', 'be', 'in', 'distress'], 0]
        ], columns=['inputs', 'labels'])

        self.assertIsInstance(convert_to_pytorch_dataset(data),(DataLoader, DataLoader))



        #pd_testing.assert_frame_equal(convert_to_pytorch_dataset(data), )

if __name__ == '__main__':
    unittest.main()
