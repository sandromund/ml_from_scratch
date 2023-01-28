letter_frequencies = {
    "english": {
        'a': 8.167,
        'b': 1.492,
        'c': 2.782,
        'd': 4.253,
        'e': 12.702,
        'f': 2.228,
        'g': 2.015,
        'h': 6.094,
        'i': 6.966,
        'j': 0.153,
        'k': 0.772,
        'l': 4.025,
        'm': 2.406,
        'n': 6.749,
        'o': 7.507,
        'p': 1.929,
        'q': 0.095,
        'r': 5.987,
        's': 6.327,
        't': 9.056,
        'u': 2.758,
        'v': 0.978,
        'w': 2.360,
        'x': 0.150,
        'y': 1.974,
        'z': 0.074
    },
    "german": {
        'a': 5.58,
        'ä': 0.54,
        'b': 1.96,
        'c': 3.16,
        'd': 4.98,
        'e': 16.93,
        'f': 1.49,
        'g': 3.02,
        'h': 4.98,
        'i': 8.02,
        'j': 0.24,
        'k': 1.32,
        'l': 3.60,
        'm': 2.55,
        'n': 10.53,
        'o': 2.24,
        'ö': 0.30,
        'p': 0.67,
        'q': 0.02,
        'r': 6.89,
        'ß': 0.37,
        's': 6.42,
        't': 5.79,
        'u': 3.83,
        'ü': 0.65,
        'v': 0.84,
        'w': 1.78,
        'x': 0.05,
        'y': 0.05,
        'z': 1.21,
    }
}

def get_text_letters(text):
if __name__ == '__main__':

    text_a = """
    In linguistics, language classification is the grouping of related languages into the same category. 
    There are two main kinds of language classification: genealogical and typological classification
    """

    text_b = """
    In der Linguistik bedeutet Sprachklassifikation die Einteilung verwandter Sprachen in dieselbe Kategorie. 
    Es gibt zwei Hauptarten der Sprachklassifizierung: die genealogische und die typologische Klassifizierung.
    """

    print("Text A is:", classify_language(text_a))
    print("Text B is:", classify_language(text_b))



