import unittest
from unittest import TestCase
import sys
import pandas as pd
import numpy as np
# import pandas.testing as pd_testing
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Dataset

import Model
from lstm_spell_classifier_w_context_onehot import get_wikipedia_text, get_bea60_data, convert_to_numpy_valdata, \
    remove_punctuation, cleanup_data, generate_N_grams, generate_N_grams_valdata, convert_to_pytorch_dataset, \
    initialize_model, \
    insert_errors, one_hot_encode_data, train_model, val_model

torch.manual_seed(0)
np.random.seed(0)


class Test_lstm_w_context_onehot(TestCase):

    def test_get_wikipedia_text(self):
        test_array = np.array(['Big Brother is a fictional character from the world of television hsitory and has been '
                               'there for ages. it needs to be seen how long it goes !',
                               'Freiburg is a city situated in the black-forest region of Germany.'])
        np.testing.assert_array_equal(get_wikipedia_text('test_file.jsonl', lower_case=False), test_array)
        test_array = np.array(['big brother is a fictional character from the world of television hsitory and has been '
                               'there for ages. it needs to be seen how long it goes !',
                               'freiburg is a city situated in the black-forest region of germany.'])
        np.testing.assert_array_equal(get_wikipedia_text('test_file.jsonl', lower_case=True), test_array)
        return


    def test_get_bea60_data(self):
        test_file_path = 'bea60k_sentences_test_file.json'

        dataset_new = {"WANT TO THANK YOU FOR": 0,
                       "In our Academy we are": 0,
                       "I was trully dissapointed by": 1,
                       "a dream becames true and": 1}

        self.assertEqual(get_bea60_data(test_file_path), dataset_new)

    def test_convert_to_numpy_valdata(self):
        dataset_old = {"WANT TO THANK YOU FOR": 0,
                       "In our Academy we are": 0,
                       "I was trully dissapointed by": 1,
                       "a dream becames true and": 1}

        dataset_new = (
            np.array(['WANT TO THANK YOU FOR', "In our Academy we are", "I was trully dissapointed by",
                      "a dream becames true and"]),
            np.array([0, 0, 1, 1])
        )

        x = convert_to_numpy_valdata(dataset_old)
        np.testing.assert_equal(x[0], dataset_new[0])
        np.testing.assert_equal(x[1], dataset_new[1])

        pass

    def test_remove_punctuation(self):
        np.testing.assert_array_equal(remove_punctuation(np.array(['Can, I be ! at office.', 'will you be there ?'])),
                                      np.array(['Can I be  at office', 'will you be there ']))

    def test_cleanup_data(self):
        pass

    def test_generate_N_grams(self):
        pass

    def test_generate_N_grams_valdata(self):
        pass

    def test_convert_to_pytorch_dataset(self):
        pass

    def test_initialize_model(self):
        pass

    def test_insert_errors(self):
        pass

    def test_one_hot_encode_data(self):
        pass

    def test_train_model(self):
        pass

    def test_val_model(self):
        pass


if __name__ == '__main__':
    unittest.main()
