import unittest
from unittest import TestCase
import numpy as np
import torch

from lstm_spell_classifier_wo_context import insert_errors, binarize, vectorize_data, get_wikipedia_words, convert_to_numpy, \
    convert_to_numpy_valdata, convert_to_pytorch_dataset, initialize_model, train_model, val_model, load_and_test_model

torch.manual_seed(0)
np.random.seed(0)

class Test_lstm_wo_context(TestCase):

    def test_get_wikipedia_words(self):

        dataset_new = {'also': 9165, 'first': 8585, 'new': 7536, 'one': 7071, 'two': 6035}
        self.assertEqual(get_wikipedia_words(file_name='test_words.json'),dataset_new)

    def test_insert_errors(self):
        pass

    def test_binarize(self):
        pass

    def test_vectorize_data(self):
        pass

    def test_convert_to_numpy(self):
        pass

    def test_convert_to_numpy_valdata(self):
        pass

    def test_convert_to_pytorch_dataset(self):
        pass

    def test_initialize_model(self):
        pass

    def test_train_model(self):
        pass

    def test_val_model(self):
        pass

    def test_load_and_test_model(self):
        pass


if __name__ == '__main__':
    unittest.main()