from unittest import TestCase
import fiction_segmentation.preprocessing.tokenize as tokenize


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
        if tokenize(test_str)!=test_result:
            self.fail()
