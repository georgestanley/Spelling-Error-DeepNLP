import unittest
import pandas as pd
import pandas.testing as pd_testing


from lstm_spell_classifier_w_context import remove_punctuation, cleanup_data, generate_N_grams


class Test_lstm_w_context(unittest.TestCase):

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation('Can, I be ! at office.'),'Can I be  at office')

    def test_cleanup_data(self):
        pd_testing.assert_frame_equal(cleanup_data(pd.DataFrame({'Freiburg, will not fall!!'},columns=['text'])),
                         pd.DataFrame({'Freiburg will not fall'},columns=['text']))

    def test_generate_N_grams(self):
        dataset_old = pd.DataFrame({'the economy will be in distress'},columns=['text'])
        dataset_new = pd.DataFrame ([
                                     [['the','economy','will','be','in']],
                                     [['economy', 'will', 'be', 'in', 'distress']]
                                     ], columns =['text'])
        pd_testing.assert_frame_equal(generate_N_grams(dataset_old,5),dataset_new)

if __name__=='__main__':
    unittest.main()
