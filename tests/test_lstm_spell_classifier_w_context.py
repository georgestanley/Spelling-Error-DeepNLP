import unittest
from unittest import TestCase
import numpy as np
import torch

from lstm_spell_classifier_w_context import remove_punctuation, cleanup_data, generate_N_grams, \
    convert_to_pytorch_dataset, get_wikipedia_text, collate_fn, parse_arguments, initialize_model, binarize2, \
    get_bea60_data, convert_to_numpy_valdata, generate_N_grams_valdata, \
    train_model, val_model, insert_errors, vectorize_data2

torch.manual_seed(0)
np.random.seed(0)


class Test_lstm_w_context(TestCase):

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

    def test_remove_punctuation(self):
        np.testing.assert_array_equal(remove_punctuation(np.array(['Can, I be ! at office.', 'will you be there ?'])),
                                      np.array(['Can I be  at office', 'will you be there ']))

    def test_cleanup_data(self):
        np.testing.assert_array_equal(cleanup_data(np.array(['Can, I be ! at office.', 'will you be there ?'])),
                                      np.array(['Can I be  at office', 'will you be there ']))

    #
    def test_generate_N_grams(self):
        dataset_old = np.array(['i hope you will be there', 'the sun is yellow and round'])
        dataset_new = ([
                           [['i', 'hope', 'you', 'will', 'be']],
                           [['hope', 'you', 'will', 'be', 'there']],
                           [['the', 'sun', 'is', 'yellow', 'and']],
                           [['sun', 'is', 'yellow', 'and', 'round']]
                       ], [0, 0, 0, 0])
        self.assertEqual(generate_N_grams(dataset_old, 5), dataset_new)

    def test_collate_fn(self):
        input = [
            (['hope', 'you', 'will', 'be', 'there'], 0),
            (['i', 'cannot', 'maze', 'it', 'tonight'], 1)
        ]
        output_x = [['hope', 'you', 'will', 'be', 'there'], ['i', 'cannot', 'maze', 'it', 'tonight']]
        output_y = [0, 1]
        self.assertEqual(collate_fn(input), (output_x, output_y))

    def test_convert_to_pytorch_dataset(self):
        input_1 = ([['hope', 'you', 'will', 'be', 'there'],
                    ['i', 'cannot', 'maze', 'it', 'tonight']], [0, 1])
        input_2 = ([['hope', 'you', 'stay', 'here', 'now'],
                    ['i', 'cannot', 'maze', 'it', 'tonight']], [0, 1])

        dataset_size = len(input_1)
        dataloader_1, dataloader_2 = convert_to_pytorch_dataset(train_data=input_1, val_data=input_2, batch_size=1)
        self.assertEqual(len(dataloader_1.dataset), dataset_size)
        self.assertEqual(len(dataloader_2.dataset), dataset_size)
        self.assertIsInstance(dataloader_1, torch.utils.data.dataloader.DataLoader)
        self.assertIsInstance(dataloader_2, torch.utils.data.dataloader.DataLoader)

    def test_initialize_model(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, criterion, optimizer = initialize_model(hidden_dim=100, hidden_layers=2, lr=0.001, device=device)
        self.assertIsInstance(model, torch.nn.parallel.DataParallel)
        self.assertIsInstance(criterion, torch.nn.modules.loss.CrossEntropyLoss)
        self.assertIsInstance(optimizer, torch.optim.Adam)

        output = model(torch.rand(1841,5,228))
        self.assertEqual(output.shape,torch.Size([1841,2]))



    def test_binarize(self):
        token = 'hyperbole'
        bin = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        torch.testing.assert_close(binarize2(token, isLabelVector=False), torch.tensor(bin))
        token = '0'
        torch.testing.assert_close(binarize2(token, isLabelVector=True), torch.tensor(int(float(token))))

    def test_vectorize_data(self):
        dataset_old = ([
                           ['i', 'hope', 'you', 'will', 'be']
                       ], [0])

        X_vec, Y_vec, X_token = vectorize_data2(dataset_old, with_error=True, shuffle=False)

        self.assertEqual(torch.Size([2, 5, 228]), X_vec.shape)
        self.assertEqual(torch.Size([2, 1]), Y_vec.shape)
        self.assertEqual((2, 5), np.shape(X_token))

        X_vec, Y_vec, X_token = vectorize_data2(dataset_old, with_error=False, shuffle=False)

        self.assertEqual(torch.Size([1, 5, 228]), X_vec.shape)
        self.assertEqual(torch.Size([1, 1]), Y_vec.shape)
        self.assertEqual((1,5), np.shape(X_token))

    def test_insert_errors(self):
        dataset_old = np.array([[
            'i', 'hope', 'you', 'will', 'be', '0'
        ]])
        dataset_new = np.array([
            ['i', 'hope', 'you', 'will', 'be', '0', ],
            ['i', 'hope', 'yu', 'will', 'be', '1.0']
        ])
        np.testing.assert_array_equal(insert_errors(dataset_old), dataset_new)

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

    def test_generate_N_grams_valdata(self):
        dataset_old = (
            np.array(['WANT TO THANK YOU FOR', "In our Academy we are"]),
            np.array([0, 1])
        )
        # new_d , labels = generate_N_grams_valdata(dataset_old)
        dataset_new = ([
                           [['WANT', 'TO', 'THANK', 'YOU', 'FOR']],
                           [['In', 'our', 'Academy', 'we', 'are']]
                       ],
                       np.array([0, 1])
        )
        self.assertEqual(generate_N_grams_valdata(dataset_old)[0], dataset_new[0])
        np.testing.assert_array_equal(generate_N_grams_valdata(dataset_old)[1], dataset_new[1])

    def test_train_model(self):
        pass

    def test_val_model(self):
        pass


if __name__ == '__main__':
    unittest.main()
