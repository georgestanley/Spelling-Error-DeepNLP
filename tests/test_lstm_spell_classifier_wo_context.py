import unittest
from unittest import TestCase
import numpy as np
import torch

import Model
from lstm_spell_classifier_wo_context import insert_errors, binarize, vectorize_data, get_wikipedia_words, convert_to_numpy, \
    convert_to_numpy_valdata, convert_to_pytorch_dataset, initialize_model, train_model, val_model

torch.manual_seed(0)
np.random.seed(0)
alph = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;'*!?`$%&(){}[]-/\\@_#"
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
        token = 'hyperbole'
        bin = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        torch.testing.assert_close(binarize(token, label=0, alph=alph )[0], torch.tensor(bin))
        spot = alph.index(token[0])
        b1,b2,b3 = bin[0:75], bin[76:151], bin[152:228]
        self.assertEqual(b1[spot],1)
        spot = alph.index(token[-1])
        self.assertEqual(b3[spot], 1)

    def test_vectorize_data(self):
        dataset_old = [
            ('february', 'January'),
            torch.tensor([0,0])
        ]
        X_vec, Y_vec, X_token = vectorize_data(dataset_old, with_error=True, shuffle=True)
        #print(str(X_vec[1].data).replace('.,','').replace('\n','').replace(' ',''))

        #t1: check X_vec shape
        self.assertEqual(X_vec.shape,torch.Size([4,228]))
        #t2 :check Y_vec data
        torch.testing.assert_close(Y_vec,torch.tensor([[0.0],[0.0],[1.0],[1.0]]))
        #t3 :check X_token data
        np.testing.assert_array_equal(X_token,np.array([['february'],
       ['January'],
       ['februar'],
       ['Januyary']]))


    def test_convert_to_numpy(self):
        dataset_old = {'january':100,
                       'february':245}

        train_data = convert_to_numpy(dataset_old)
        np.testing.assert_array_equal(train_data[0], np.array(['january','february']))
        np.testing.assert_array_equal(train_data[1],np.array([0,0]))


    def test_convert_to_numpy_valdata(self):
        dataset_old = {'january': 0,
                       'february': 1}
        train_data = convert_to_numpy_valdata(dataset_old)
        np.testing.assert_array_equal(train_data[0], np.array(['january', 'february']))
        np.testing.assert_array_equal(train_data[1], np.array([0, 1]))
        pass

    def test_convert_to_pytorch_dataset(self):
        train_words = np.array(['january','february','janury','februiary'])
        train_labels = np.array([0,0,1,1])
        train_data = (train_words,train_labels)

        val_words = np.array(['Thanks', 'Dhanks'])
        val_labels = np.array([0, 1])
        val_data = (val_words,val_labels)

        train_loader, val_loader = convert_to_pytorch_dataset(train_data,val_data, batch_size=2)
        self.assertEqual(len(train_loader.dataset), len(train_words) )
        self.assertEqual(len(val_loader.dataset), len(val_words))
        self.assertIsInstance(train_loader, torch.utils.data.dataloader.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.dataloader.DataLoader)

    def test_initialize_model(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, criterion, optimizer = initialize_model(n_hidden_layers=2,hidden_dim=256,lr=0.001, device=device)
        self.assertIsInstance(model, Model.LSTMModel)
        self.assertIsInstance(criterion, torch.nn.modules.loss.CrossEntropyLoss)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        pass

    def test_train_model(self):

        pass

    def test_val_model(self):
        pass

    def test_load_and_test_model(self):

        pass


if __name__ == '__main__':
    unittest.main()