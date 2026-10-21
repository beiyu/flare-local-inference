"""Exact tie handling shared by ordinary and speculative zero-temperature sampling."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ignition.components.greedy import PreparedGreedyDistribution


@unittest.skipIf(torch is None, "requires PyTorch")
class GreedyDistributionTests(unittest.TestCase):
    def test_mixed_rows_choose_lowest_maximum_and_preserve_stochastic_probabilities(self):
        scores = torch.tensor([[1., 5., 5., -torch.inf], [2., 2., 2., 1.], [-3., -1., -1., -2.]])
        probabilities = torch.tensor([[.1, .4, .4, .1], [.3, .3, .3, .1], [.1, .4, .4, .1]])
        expected = torch.tensor([[0., 1., 0., 0.], [.3, .3, .3, .1], [0., 1., 0., 0.]])
        output = torch.empty((3, 1), dtype=torch.int64)
        operation = PreparedGreedyDistribution(output, 4)
        pointer = probabilities.data_ptr()
        operation(scores, probabilities, torch.tensor([True, False, True]))
        torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
        self.assertEqual(probabilities.data_ptr(), pointer)
        self.assertEqual(output[:, 0].tolist(), [1, 0, 1])

    def test_signed_zero_and_changing_rows_do_not_retain_a_previous_argmax(self):
        indices = torch.empty((2, 1), dtype=torch.int64)
        operation = PreparedGreedyDistribution(indices, 4)
        scores = torch.tensor([[-0., 0., -0., -1.], [0., 0., 1., 1.]])
        probabilities = torch.full((2, 4), .25)
        greedy = torch.ones(2, dtype=torch.bool)
        operation(scores, probabilities, greedy)
        self.assertEqual(probabilities.argmax(-1).tolist(), [0, 2])
        scores[:, 3] = 2
        operation(scores, probabilities, greedy)
        self.assertEqual(probabilities.argmax(-1).tolist(), [3, 3])
        self.assertEqual(probabilities.sum(-1).tolist(), [1., 1.])


if __name__ == "__main__":
    unittest.main()
