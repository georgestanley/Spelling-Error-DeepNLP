import unittest
from unittest import TestCase
from application.Model import LSTMModel, LSTMModelForOneHotEncodings
import torch

torch.manual_seed(0)


class Test_models(TestCase):

    def test_LSTMModel(self):
        device = 'cuda'
        X = torch.rand(100, 5, 228)
        model = LSTMModel(input_dim=228, hidden_dim=512, layer_dim=2, output_dim=2, device=device).to(device)
        out = model(X.to(device))
        self.assertEqual(out.shape, torch.Size([100, 2]))
        self.assertAlmostEqual(torch.sum(out).item(), -4.7564, delta=0.001)
        pass

    def test_LSTMModelForOneHotEncodings(self):  #
        device = 'cuda'
        X = torch.rand(61, 60, 77)
        x_bar = torch.rand(61)
        model = LSTMModelForOneHotEncodings(input_dim=77, hidden_dim=512, layer_dim=2, output_dim=2, device=device).to(
            device)
        out = model(X.to(device), x_bar.type(torch.LongTensor).to(device=device))
        self.assertEqual(out.shape, torch.Size([61, 2]))
        self.assertAlmostEqual(torch.sum(out).item(), -3.5607, delta=0.001)
        pass




if __name__ == '__main__':
    unittest.main()
