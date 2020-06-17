import math

from src.misc import cosine_similarity

member_vector = lambda word, sentence: 1 if (word in sentence) else 0


def term_frequency(term, sentence):
    return sentence.count(term) / len(sentence)


def inverse_data_frequency(term, data):
    m = sum([sentence.count(term) for sentence in data])
    if m == 0:
        return 0
    return math.log(len(data) / m)


def TF_IDF(term, sentence, data, corpus):
    tf = term_frequency(term, sentence)
    idf = inverse_data_frequency(term, data)
    return tf * idf


def preprocess(sentence):
    word_vector = [w.upper() for w in sentence.split(' ')]
    for i in range(len(word_vector)):
        word_vector[i] = "".join([char for char in word_vector[i]
                                  if char.isalpha()])
    return word_vector


def corpus_vector(s, corpus):
    return [member_vector(v, s) for v in corpus]


def vector_TF_IDF(w, data, corpus, v_bin):
    v = []
    for i in range(len(v_bin)):
        if v_bin[i] == 1:
            v += [TF_IDF(corpus[i], w, data, corpus)]
        else:
            v += [0.0]
    return v


if __name__ == '__main__':
    corpus = ['THE', 'CAR', 'TRUCK', 'IS', 'DRIVEN', 'ON', 'THE', 'ROAD', 'HIGHWAY']

    s1 = "The car is driven on the road."
    s2 = "The truck is driven on the highway."
    s3 = 'The Truck is on the highway.'
    s4 = 'The Truck on highway.'

    data = [preprocess(s) for s in [s1, s2, s3, s4]]

    data_bin = [corpus_vector(s, corpus) for s in data]

    data_v = [vector_TF_IDF(data[i], data, corpus, data_bin[i]) for i in range(len(data))]

    print(cosine_similarity(data_v[1], data_v[2]))
