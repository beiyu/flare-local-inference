"""Exact argmax distributions with lowest-token-ID tie breaking."""

import torch

from ignition.preparation.tensor_metadata import record_numeric_tensor


class PreparedGreedyDistribution:
    """Replace only zero-temperature rows after penalties and grammar masking.

    Top-k threshold filtering preserves tied maxima and therefore cannot express
    greedy decoding alone. Argmax selects the first vocabulary index explicitly.
    The integer reduction output and vocabulary IDs are retained through capture.
    """

    def __init__(self, indices, vocabulary, *, vocab_ids=None):
        self.indices = indices
        self.vocab_ids = (torch.arange(vocabulary, device=indices.device)
                          if vocab_ids is None else vocab_ids)
        record_numeric_tensor(self.vocab_ids)

    def __call__(self, scores, probabilities, greedy):
        torch.argmax(scores, dim=-1, keepdim=True, out=self.indices)
        probabilities.copy_(torch.where(greedy[:, None], self.vocab_ids[None, :] == self.indices,
                                        probabilities))
        return probabilities
