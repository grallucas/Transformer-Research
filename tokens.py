import numpy as np
import re

def tokenize_file(file_path):
    with open(file_path, encoding="utf-8") as file:
        corpus_text = file.read()

    # SPLIT to isolate alphabetic strings, digit strings, and punctuation
    word_split = lambda text: re.findall(r' ?[^\W\d_]+|[\s\S]', text)
    WORDS = word_split(corpus_text)

    WORD_TO_ID = {
        word: index
        for index, word in enumerate(set(WORDS))
    }

    ID_TO_WORD = {
        index: word
        for word, index in WORD_TO_ID.items()
    }

    VOCAB_SIZE = len(WORD_TO_ID)
    UNKNOWN_ID = WORD_TO_ID.get("?", 0)

    def encode(text):
        token_ids = [
            WORD_TO_ID.get(word, UNKNOWN_ID)
            for word in word_split(text)
        ]
        return np.asarray(token_ids, dtype=np.int64)

    def decode(token_ids):
        return " ".join(
            ID_TO_WORD[int(token_id)]
            for token_id in token_ids
        )

    tokens = encode(corpus_text)

    print('Vocabulary size:', VOCAB_SIZE)
    print('Corpus length:', len(tokens))
    print('Example tokens:')
    for token in tokens[:15]:
        print("'", decode([token]), "'", end=", ", sep='')

    return encode, decode, VOCAB_SIZE, tokens