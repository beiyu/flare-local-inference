"""E3 segment geometry at the complete 288-entry pointer-table boundary."""

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ignition.components.exl3_prefill import grouped_tables


@unittest.skipIf(torch is None, "requires PyTorch")
class PrefillTableTests(unittest.TestCase):
    def tables(self, counts):
        counts = torch.tensor(counts, dtype=torch.int64)
        slots = max(int(counts.sum()) + 30, 1)
        tokens = torch.arange(slots, dtype=torch.int64)
        weights = torch.linspace(.01, .5, slots, dtype=torch.float16)
        table = grouped_tables(counts, tokens, weights, torch.arange(slots),
            torch.arange((slots + 63) // 64 + 288))
        return tokens, weights, table

    def test_first_last_experts_empty_gaps_and_exact_thin_threshold(self):
        counts = [0] * 288
        counts[0], counts[17], counts[287] = 33, 32, 65
        tokens, weights, (segments, row_expert, row_token, row_weight, num_rows, num_segs) = self.tables(counts)
        self.assertEqual((num_rows.item(), num_segs.item()), (98, 3))
        self.assertEqual(list(zip(*(value[:3].tolist() for value in segments))),
            [(0, 0, 33), (287, 33, 64), (287, 97, 1)])
        self.assertEqual(row_expert[:98].tolist(), [0] * 33 + [287] * 65)
        torch.testing.assert_close(row_token[:98], torch.cat((tokens[:33], tokens[65:130])), rtol=0, atol=0)
        torch.testing.assert_close(row_weight[:98], torch.cat((weights[:33], weights[65:130])), rtol=0, atol=0)

    def test_all_288_fat_experts_fit_capacity_and_empty_or_thin_routes_do_not_enter_e3(self):
        _, _, (segments, row_expert, _, _, rows, segs) = self.tables([33] * 288)
        self.assertEqual((rows.item(), segs.item()), (33 * 288, 288))
        self.assertEqual(segments[0][:288].tolist(), list(range(288)))
        self.assertEqual(segments[1][:288].tolist(), list(range(0, 33 * 288, 33)))
        self.assertTrue(torch.equal(segments[2][:288], torch.full((288,), 33, dtype=torch.int32)))
        self.assertEqual(row_expert[:33 * 288].tolist(), [expert for expert in range(288) for _ in range(33)])
        for count in (0, 32):
            _, _, (*_, rows, segs) = self.tables([count] * 288)
            self.assertEqual((rows.item(), segs.item()), (0, 0))


if __name__ == "__main__":
    unittest.main()
