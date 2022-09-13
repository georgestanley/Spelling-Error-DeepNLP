import unittest
from unittest import TestCase
import numpy as np
# import pandas.testing as pd_testing
import torch.utils.data
import application.Model as Model

from application.lstm_spell_classifier_w_context_onehot import get_wikipedia_text, get_bea60_data, convert_to_numpy_valdata, \
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
        np.testing.assert_array_equal(get_wikipedia_text('application//tests//test_file.jsonl', lower_case=False), test_array)
        test_array = np.array(['big brother is a fictional character from the world of television hsitory and has been '
                               'there for ages. it needs to be seen how long it goes !',
                               'freiburg is a city situated in the black-forest region of germany.'])
        np.testing.assert_array_equal(get_wikipedia_text('application//tests//test_file.jsonl', lower_case=True), test_array)
        return

    def test_get_bea60_data(self):
        test_file_path = 'application//tests//bea60k_sentences_test_file.json'

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
        np.testing.assert_array_equal(cleanup_data(np.array(['Can, I be ! at office.', 'will you be there ?'])),
                                      np.array(['Can I be  at office', 'will you be there ']))

    def test_generate_N_grams(self):
        dataset_old = np.array(['i hope you will be there',
                                'the sun is yellow and round'])
        dataset_new = (np.array([
            'i hope you will be',
            'hope you will be there',
            'the sun is yellow and',
            'sun is yellow and round']),
                       np.array([0, 0, 0, 0]))
        np.testing.assert_array_equal(generate_N_grams(dataset_old)[0], dataset_new[0])
        np.testing.assert_array_equal(generate_N_grams(dataset_old)[1], dataset_new[1])

    def test_generate_N_grams_valdata(self):
        dataset_old = (
            np.array(['WANT TO THANK YOU FOR', "In our Acadeemy we are"]),
            np.array([0, 1])
        )
        # new_d , labels = generate_N_grams_valdata(dataset_old)
        dataset_new = (np.array([
            'WANT TO THANK YOU FOR',
            'In our Acadeemy we are'
        ]),
                       np.array([0, 1])
        )
        # self.assertEqual(generate_N_grams_valdata(dataset_old)[0], dataset_new[0])
        np.testing.assert_array_equal(generate_N_grams_valdata(dataset_old)[0], dataset_new[0])
        np.testing.assert_array_equal(generate_N_grams_valdata(dataset_old)[1], dataset_new[1])

    def test_convert_to_pytorch_dataset(self):
        input_1 = (np.array(['hope you will be there',
                    'i cannot maze it tonight']), np.array([0, 1]))
        input_2 = (np.array(['hope you will be there',
                    'i cannot maze it tonight']), np.array([0, 1]))

        dataset_size = len(input_1)
        dataloader_1, dataloader_2 = convert_to_pytorch_dataset(train_data=input_1, val_data=input_2, batch_size=1)
        self.assertEqual(len(dataloader_1.dataset), dataset_size)
        self.assertEqual(len(dataloader_2.dataset), dataset_size)
        self.assertIsInstance(dataloader_1, torch.utils.data.dataloader.DataLoader)
        self.assertIsInstance(dataloader_2, torch.utils.data.dataloader.DataLoader)

        for i, data in enumerate(dataloader_1):
            if i==0:
                np.testing.assert_array_equal(data[0],np.array(['hope you will be there']))
                torch.testing.assert_close(data[1],torch.tensor([0]))
            if i==1:
                np.testing.assert_array_equal(data[0], np.array(['i cannot maze it tonight']))
                torch.testing.assert_close(data[1], torch.tensor([1]))

        for i, data in enumerate(dataloader_2):
            if i==0:
                np.testing.assert_array_equal(data[0],np.array(['i cannot maze it tonight','hope you will be there',]))
                torch.testing.assert_close(data[1],torch.tensor([1,0]))
            if i==1:
                self.assert_(True,'Dataset Size Exceeded. Check !')


    def test_initialize_model(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, criterion, optimizer = initialize_model(hidden_dim=100, hidden_layers=2, lr=0.001, device=device)
        self.assertIsInstance(model, torch.nn.parallel.DataParallel)
        self.assertIsInstance(criterion, torch.nn.modules.loss.CrossEntropyLoss)
        self.assertIsInstance(optimizer, torch.optim.Adam)

        output = model(torch.rand(61, 60,77), [10]*61)
        self.assertEqual(output.shape, torch.Size([61, 2]))

    def test_insert_errors(self):
        dataset_old =(
            'i hope you will be','I will make something today'
        )
        dataset_new = ['i hope you will be', 'I will make something today', 'i hope yu will be', 'I will mak '
                                                                                                 'something today']

        self.assertEqual(insert_errors(dataset_old)[0], dataset_new)
        self.assertEqual(insert_errors(dataset_old)[1], [0,0,1,1])

    def test_one_hot_encode_data(self):
        dataset_old = (
            'x x Stanley x x', 'y y Stephin y y'
        )
        labels = torch.tensor([0, 0])
        op,l,arr_len = one_hot_encode_data(dataset_old,with_error=True,labels=labels, shuffle=True, maxlen=60)
        torch.testing.assert_close(op[0][0][49],torch.tensor(1)) #x
        torch.testing.assert_close(op[0][1][76],torch.tensor(1)) #(space)
        torch.testing.assert_close(op[0][2][49],torch.tensor(1)) #x
        self.assertEqual(arr_len,[14,14,15,13])
        self.assertEqual(op.shape,torch.Size([4,60,77]))
        pass

    def test_train_model(self):
        pass

    def test_val_model(self):
        pass


if __name__ == '__main__':
    unittest.main()
