from application.utils.utils import str2bool, int2char, get_rand01, get_rand123, check_dir, get_logger, \
    plot_graphs, f1_score_manual

import unittest
import shutil
import os
from unittest import TestCase
from pathlib import Path


class Test_lstm_w_context_onehot(TestCase):
    def test_str2bool(self):
        self.assertEqual(str2bool('true'), True)
        self.assertEqual(str2bool('false'), False)
        pass

    def test_int2char(self):
        self.assertEqual(int2char(5), 'f')
        pass

    def test_get_rand01(self):
        #self.assertEqual(get_rand01(), 1)
        self.assertIn(get_rand01(),[0,1])

    def test_get_rand123(self):
        self.assertIn(get_rand123(),[1,2,3])
        pass

    def test_check_dir(self):
        dirpath = Path('temp_dir')
        if dirpath.exists():
            shutil.rmtree('temp_dir')  # remove the directory if exists. Only for testing purpose
        self.assertEqual(check_dir('temp_dir'), 'temp_dir')
        pass

    def test_get_logger(self):
        check_dir("logdir")
        l = get_logger('logdir', 'test_logger')
        self.assertEqual(os.path.exists(l.handlers[0].baseFilename), True)

    def test_plot_graphs(self):
        check_dir('logdir')
        l = get_logger('logdir', 'test_logger')
        model_folder = check_dir('test_model_folder')
        try:
            plot_graphs(2, model_folder, l, [0, 0.022, 0.333], [0, 0.111, 0.333], [0, 34, 45], [0, 0.34, 0.22])
        except Exception as e:
            raise AssertionError()

    def test_f1_score_manual(self):
        tn = 55
        fp = 5
        fn = 10
        tp = 30
        f1_calculated = 0.799
        self.assertAlmostEqual(f1_score_manual(tn, fp, fn, tp), f1_calculated, delta=0.001)
        pass


def cleanup_folders():
    shutil.rmtree('logdir')
    shutil.rmtree('test_model_folder')


if __name__ == '__main__':
    unittest.main()
    cleanup_folders()
