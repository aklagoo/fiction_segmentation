from unittest import TestCase
from fiction_segmentation.preprocessing import tokenize


class TestTokenize(TestCase):
    def test_tokenize(self):
        # Test data
        test_str = '''Meh; meh. Boo!
        Tink tonk. Bruh
        '''
        test_result = [
            ['Meh', ';'],
            ['meh', '.'],
            ['Boo', '!'],
            ['Tink', 'tonk', '.'],
            ['Bruh']
        ]
        self.assertEqual(tokenize(test_str) == test_result, True)
