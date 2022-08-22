from application.utils.utils import str2bool, save_in_log, int2char, get_rand01, get_rand123, check_dir, get_logger, plot_graphs, f1_score_manual

import unittest
from unittest import TestCase

class Test_lstm_w_context_onehot(TestCase):
    def test_str2bool(self):
        self.assertEqual(str2bool('true'),True)
        self.assertEqual(str2bool('false'), False)
        pass

    def test_int2char(self):
        """Given an integer between 0 and 25, returns the corresponding english alphabet of that position. Used in the
        insert_errors function"""
        self.assertEqual(int2char(5),'f')
        pass

    def test_get_rand01(self):

        self.assertEqual()
        pass


    def test_check_dir(self):
        pass

    def test_get_logger(self):

        pass

    def test_plot_graphs(self):
        pass

    def test_f1_score_manual(self):
        tn = 55
        fp = 5
        fn = 10
        tp = 30
        precision = tp / (tp+fp)
        recall = tp / (tp + fn)
        f1_calculated= 0.799
        self.assertAlmostEqual(f1_score_manual(tn,fp,fn,tp), f1_calculated, delta=0.001)
        pass





if __name__ == '__main__':
    unittest.main()



