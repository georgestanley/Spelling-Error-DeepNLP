import unittest
from unittest import TestCase
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

from application.app import convert_to_numpy_valdata, generate_N_grams, generate_N_grams_onehot, \
    evaluate_lstm_context_semi_character, evaluate_lstm_context_one_hot, evaluate_lstm_wo_context

from application.lstm_spell_classifier_w_context import initialize_model as initialize_semi_character_w_context
from application.lstm_spell_classifier_w_context_onehot import initialize_model as initialize_one_hot
from application.lstm_spell_classifier_wo_context import initialize_model as initialize_semi_character_wo_context

###### LOAD THE MODELS ######
device = 'cuda'
PATH_SEMI_CHARACTER = "application//tests//models//semi_character_w_context.pth"
model_semi_character_w_context, _, _ = initialize_semi_character_w_context(hidden_dim=512, hidden_layers=2,
                                                                           lr=0.001,
                                                                           device=device)
model_semi_character_w_context.load_state_dict(torch.load(PATH_SEMI_CHARACTER))
model_semi_character_w_context.eval()

######
PATH_ONE_HOT = "application//tests//models//onehot_w_context.pth"

device = 'cuda'
model_one_hot, _, _ = initialize_one_hot(hidden_dim=512, hidden_layers=2,
                                         lr=0.001,
                                         device=device)
model_one_hot.load_state_dict(torch.load(PATH_ONE_HOT))
model_one_hot.eval()

######
PATH_WO_CONTEXT = "application//tests//models//semi_character_wo_context.pth"
device = 'cuda'
model_semi_character_wo_context, criterion, _ = initialize_semi_character_wo_context(n_hidden_layers=2,
                                                                                     hidden_dim=1024,
                                                                                     lr=0.01,
                                                                                     device=device)
model_semi_character_wo_context.load_state_dict(torch.load(PATH_WO_CONTEXT))
model_semi_character_wo_context.eval()


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

    def test_generate_n_grams_onehot(self):
        data_input = (
            np.array(['We can develop a better system']),
            np.array([0])
        )
        data_output, _ = generate_N_grams_onehot(data_input)
        data_output_expected = np.array(['* * We can develop',
                                         '* We can develop a',
                                         'We can develop a better',
                                         'can develop a better system',
                                         'develop a better system *',
                                         'a better system * *'])
        np.testing.assert_equal(data_output, data_output_expected)

    def test_evaluate_lstm_context_semi_character(self):
        data_input = 'We can develop a better system'
        data_output = evaluate_lstm_context_semi_character(model_semi_character_w_context, data_input)
        self.assertEqual(data_output, [])

        data_input = 'We can develp a better system'
        data_output = evaluate_lstm_context_semi_character(model_semi_character_w_context, data_input)
        self.assertEqual(data_output, ['develp'])

    def test_evaluate_lstm_context_onehot(self):
        data_input = 'We can develop a better system'
        data_output = evaluate_lstm_context_one_hot(model_one_hot, data_input)
        self.assertEqual(data_output, ['We'])

        data_input = 'We can develp a better system'
        data_output = evaluate_lstm_context_one_hot(model_one_hot, data_input)
        self.assertEqual(data_output, ['We', 'can', 'develp'])

        pass

    def test_evaluate_lstm_wo_context(self):
        data_input = 'We can develop a better system'
        data_output = evaluate_lstm_wo_context(model_semi_character_wo_context, data_input)
        self.assertEqual(data_output, [])

        data_input = 'We can develp a better system'
        data_output = evaluate_lstm_wo_context(model_semi_character_wo_context, data_input)
        self.assertEqual(data_output, ['develp'])

        pass

    def test_update_output(self):
        "Due to the design limitation of the dash app, we can't create a test function for it."
        pass


if __name__ == '__main__':
    unittest.main()
