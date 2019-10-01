import nltk


def tokenize(text: str) -> list:
    """Splits text into lines, sentences and words"""
    tokenized_text = []
    for line in text.split('\n'):
        for sentence in nltk.sent_tokenize(line):
            # Split the sentence if a semi-colon is encountered without removing it
            subsentences = [subsentence + ';' for subsentence in sentence.split(';') if subsentence]
            subsentences[-1] = subsentences[-1][:-1]

            # Word tokenize each subsentence
            subsentences = [nltk.word_tokenize(subsentence) for subsentence in subsentences]
            tokenized_text.extend(subsentences)

    return tokenized_text


def stem(text):
    """Stems each word in the tokenized text"""
    ps = nltk.PorterStemmer()

    stemmed_text = []
    for sentence in text:
        stemmed_sentence = []
        for word in sentence:
            stemmed_sentence.append(ps.stem(word.lower()))
        stemmed_text.append(stemmed_sentence)

    return stemmed_text
