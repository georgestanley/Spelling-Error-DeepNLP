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

        dataset_old = np.array([
            ['february',0],
            ['January', 0],
            ['boy',0]
        ])
        dataset_new = np.array([
                                ['february', '0'],
                                ['January' ,'0'],
                                ['boy','0'], # no error word as len <= 3
                                ['febuary' ,'1.0'],
                                ['Jhanuary' ,'1.0']
        ])
        np.testing.assert_array_equal(insert_errors(dataset_old), dataset_new)


    def test_binarize(self):
        pass

    def test_vectorize_data(self):
        dataset_old = [
            ('february', 'January'),
            torch.tensor([0,0])
        ]
        X_vec, Y_vec, X_token = vectorize_data(dataset_old, with_error=True, shuffle=True)
        print(str(X_vec[1].data).replace('.,','').replace('\n','').replace(' ',''))
        print(X_vec[2])
        print(X_vec[3])

        #t1: check X_vec shape
        #t2 :check Y_vec data
        #t3 :check X_token data
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