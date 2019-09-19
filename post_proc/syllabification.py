import codecs

def syllabification(word):
    vowels = ["u", "e", "a", "o", "i"]
    word = list(word)
    syllable = 0
    for i, vowel in enumerate(word):
        if vowel in vowels:
            syllable = 1
        if i + 2 < len(word) and word[i + 2] in vowels and syllable:
            word.insert(i + 1, "-")
            syllable = 0
    return ("".join(word))

