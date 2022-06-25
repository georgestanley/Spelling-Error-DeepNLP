import unittest
from unittest import TestCase
import sys
import pandas as pd
import numpy as np
#import pandas.testing as pd_testing
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Dataset

import Model
from lstm_spell_classifier_w_context import remove_punctuation, cleanup_data, generate_N_grams, \
    convert_to_pytorch_dataset, get_wikipedia_text, collate_fn, parse_arguments, initialize_model


class Test_lstm_w_context(TestCase):

    def test_get_wikipedia_text(self):
        test_array = np.array('big brother is a fictional character')
        np.testing.assert_array_equal(get_wikipedia_text('test_file.jsonl'), test_array)

    def test_remove_punctuation(self):
         np.testing.assert_array_equal(remove_punctuation(np.array(['Can, I be ! at office.','will you be there ?'])),
                          np.array(['Can I be  at office','will you be there ']))

    def test_cleanup_data(self):
         #pd_testing.assert_frame_equal(cleanup_data(pd.DataFrame({'Freiburg, will not fall!!'}, columns=['text'])),
         #                              pd.DataFrame({'Freiburg will not fall'}, columns=['text']))
         np.testing.assert_array_equal(remove_punctuation(np.array(['Can, I be ! at office.', 'will you be there ?'])),
                                       np.array(['Can I be  at office', 'will you be there ']))

    #
    def test_generate_N_grams(self):
        dataset_old = np.array(['i hope you will be there', 'the sun is yellow and round'])
        dataset_new = ([
            [['i','hope','you','will','be']],
             [['hope','you','will','be','there']],
             [['the','sun','is','yellow','and']],
             [['sun','is','yellow','and','round']]
        ], [0,0,0,0])
        self.assertEqual(generate_N_grams(dataset_old, 5), dataset_new)

    def test_collate_fn(self):
        input = [
            (['hope','you','will','be','there'],0),
            (['i','cannot','maze','it','tonight'],1)
        ]
        output_x = [['hope','you','will','be','there'],['i','cannot','maze','it','tonight'] ]
        output_y = [0, 1]
        self.assertEqual(collate_fn(input),(output_x,output_y))

    def test_convert_to_pytorch_dataset(self):
        input_1 = ([['hope','you','will','be','there'],
                  ['i','cannot','maze','it','tonight']], [0,1])
        input_2 = ([['hope','you','stay','here','now'],
                  ['i','cannot','maze','it','tonight']], [0,1])

        args = parse_arguments()
        dataset_size = 2
        dataloader_1, dataloader_2 = convert_to_pytorch_dataset(input_1,input_2,args)
        self.assertEqual(len(dataloader_1.dataset),dataset_size)
        self.assertEqual(len(dataloader_2.dataset),dataset_size)
        self.assertIsInstance(dataloader_1,torch.utils.data.dataloader.DataLoader)
        self.assertIsInstance(dataloader_2, torch.utils.data.dataloader.DataLoader)

    def test_initialize_model(self):
        args = parse_arguments()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, criterion, optimizer = initialize_model(args, device)
        self.assertIsInstance(model,Model.LSTMModel)
        self.assertIsInstance(criterion,torch.nn.modules.loss.CrossEntropyLoss)
        self.assertIsInstance(optimizer, torch.optim.adam.Adam)

    def test_binarize(self):

        pass


if __name__ == '__main__':
    unittest.main()
