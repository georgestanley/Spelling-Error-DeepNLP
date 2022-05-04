import unittest
import numpy as np
from lstm_spell_classifier_wo_context import insert_errors

def test_insert_errors():
    data_arr = np.ndarray([['stringa',0],['stringb',0]])
    assert isinstance((insert_errors(data_arr)),np.ndarray)