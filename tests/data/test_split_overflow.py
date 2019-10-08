from unittest import TestCase
from fiction_segmentation.data import split_overflow


class TestSplitOverflow(TestCase):
    def test_split_overflow(self):
        # Test data
        test_inp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        test_prev_len = 3
        test_sample_size = 5
        test_first = [1, 2]
        test_ready = [
            [3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12]
        ]
        test_last = [13, 14, 15, 16]

        # Test if output is correct
        first, ready, last = split_overflow(test_inp, test_prev_len, test_sample_size)

        err = []
        if first != test_first:
            err.append("first:\n{0}\ndoes not match test_first:\n{1}".format(
                str(first), str(test_first)))
        if ready != test_ready:
            err.append("ready:\n{0}\ndoes not match test_ready:\n{1}".format(
                str(ready), str(test_ready)))
        if last != test_last:
            err.append("last:\n{0}\ndoes not match test_last:\n{1}".format(
                str(last), str(test_last)))

        assert not err, "Errors found:\n{}".format("\n".join(err))
