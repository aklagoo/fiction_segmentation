from unittest import TestCase
from fiction_segmentation.preprocessing import stem


class TestStem(TestCase):
    def test_stem(self):
        test_sent = [
            ['It', 'was', 'working', '.'],
            ['His', 'bloody', 'toe', 'was', 'hurting']
        ]
        test_output = [
            ['it', 'wa', 'work', '.'],
            ['hi', 'bloodi', 'toe', 'wa', 'hurt']
        ]
        self.assertEqual(stem(test_sent) == test_output, True)
