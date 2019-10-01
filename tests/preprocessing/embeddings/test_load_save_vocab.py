from unittest import TestCase
from fiction_segmentation.preprocessing.embeddings import save_vocab, load_vocab


class TestLoadSaveVocab(TestCase):
    def test_load_save_vocab(self):
        # Sample data
        test_word2idx = {
            'hell': 0,
            'bell': 1,
            'yell': 2
        }
        test_idx2word = ['hell', 'bell', 'yell']
        save_vocab(test_word2idx, 'temp')
        out_idx2word, out_word2idx = load_vocab('temp')

        err = []
        if out_idx2word != test_idx2word:
            err.append("out_idx2word:\n{0}\ndoes not match test_idx2word:\n{1}".format(
                str(out_idx2word), str(test_idx2word)))
        if out_word2idx != test_word2idx:
            err.append("out_word2idx:\n{0}\ndoes not match test_word2idx:\n{1}".format(
                str(out_word2idx), str(test_word2idx)))

        assert not err, "Errors found:\n{}".format("\n".join(err))
