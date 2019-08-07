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

def test_syllabification(file_in, file_out):
    errors = []
    with codecs.open(file_in, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line.startswith(u'#'):
                line = line.split()
                if line[1] == 'sent_id':
                    sent_id = line[3]
                continue
            line = line.split()
            original_word = line[3]
            word = syllabification(original_word.replace("-", ""))
            if word != original_word:
                errors.append([sent_id, line[0], word, original_word])

    with codecs.open(file_out, encoding='utf-8', mode='w') as f:
        for i, _ in enumerate(errors):
            f.write('sent_id:{} word:{}\nk:{}\np:{}\n\n'.format(errors[i][0], errors[i][1], errors[i][3], errors[i][2]))


if __name__ == '__main__':
    test_syllabification('../data/HaaretzOrnan_annotated.txt', 'errors.txt')