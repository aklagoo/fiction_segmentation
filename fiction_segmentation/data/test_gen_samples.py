from unittest import TestCase
from fiction_segmentation.data import gen_samples


class TestGen_samples(TestCase):
    def test_gen_samples(self):
        # Test data
        test_content = [
            [1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28]
        ]
        test_output = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 0, 0, 0],
            [18, 19, 20, 0, 0],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 0, 0]
        ]
        test_sample_size = 5

        # Get output
        output = gen_samples(test_content, test_sample_size)

        err = []
        if output != test_output:
            err.append("output:\n{0}\ndoes not match test_output:\n{1}".format(
                str(output), str(test_output)))

        assert not err, "Errors found:\n{}".format("\n".join(err))
