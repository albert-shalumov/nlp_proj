import codecs



cache = set()

'''
Performs character-wise conversion from hebrew unicode script to characters suggested by Uzi Ornan

Input: 
w - hebrew word

Return:
word in Ornan convention
'''
def Hebrew2Ornan(w):
    # Used https://unicode-table.com as reference
    char_lut = {
        u'\u05d0':u'\u02c0', u'\u05d1':u'b',
        u'\u05d2':u'g', u'\u05d3':u'd',
        u'\u05d4':u'h', u'\u05d5':u'w',
        u'\u05d6':u'z', u'\u05d7':u'\u1e25',
        u'\u05d8':u'\u1e6d', u'\u05d9':u'y',
        u'\u05da':u'k', u'\u05db':u'k',
        u'\u05dc':u'l', u'\u05dd':u'm',
        u'\u05de':u'm', u'\u05df':u'n',
        u'\u05e0':u'n', u'\u05e1':u's',
        u'\u05e2':u'\u02c1', u'\u05e3':u'p',
        u'\u05e4':u'p', u'\u05e5':u'\u00e7',
        u'\u05e6':u'\u00e7', u'\u05e7':u'q',
        u'\u05e8':u'r', u'\u05e9':u'\u0161',
        u'\u05ea':u't'}
    res = u''
    global cache

    # Special cases that interfere with one to one conversion
    if w == u'...':
        res = u'\u2026'
    else:
        for ch in w:
            cache.add(ch)
            res += char_lut[ch] if ch in char_lut else ch
    return res

# Not in use
def BreakLines(text):
    prev_ch = u''
    quot = False
    parent = False
    new_text = u''
    for ch in text:
        if ch == '"' and prev_ch in [u'', u' ']:
            quot = not quot
        if ch == u'(':
            parent = True
        if ch == u')':
            parent = False
        new_text += ch
        if not parent and not quot and ch in [u'?', u'!', u'.']:
            new_text += u'\n'
            prev_ch = u''
            continue
        prev_ch = ch

    return new_text.splitlines()

def IsSkipWord(word:str):
    if word.isdecimal():
        return True
    if len(word)==1 and word in [u',',u'\'',u'"',u'!',u'.',u'?', u')',u'(',u'-',u';',u':',]:
        return True
    if u'"' in word:
        return True

'''
Converts hebrew text in unicode to expected word-wise structure for syllabification.
Note: this function is somewhat specific for text for MILA Haaretz text.
Format:
for each sentence:
    # sent_id = sentence_index
    # text = sentence_hebrew_unicode
    for each word:
        word_index word_hebrew word_ornan word_ornan
    \n

Input: 
file_in - hebrew text file
file_out - output file

Return:
None
'''
def ConvertRaw(file_in, file_out):
    with codecs.open(file_in, encoding='utf-8') as f:
        lines = f.readlines()
        hebrew_text = list()
        # remove line numbers and "empty" lines
        for line in lines:
            split_line = line.split()
            if split_line[0].isnumeric():
                split_line = split_line[1:]
            else:
                split_line = split_line[2:]
            if len(split_line) == 0:
                continue
            elif len(split_line) == 1 and split_line[0] == '*':
                continue
            hebrew_text += split_line

    # Reduce end sentence ambiguity
    hebrew_text = u' '.join(hebrew_text)
    hebrew_text = hebrew_text.replace(u'...', u'\u2026')

    # split into sentences
    hebrew_text = hebrew_text.replace(u'?', u'?\n')
    hebrew_text = hebrew_text.replace(u'.', u'.\n')
    hebrew_text = hebrew_text.replace(u'!', u'!\n')
    hebrew_text = hebrew_text.splitlines()
    #hebrew_text = BreakLines(hebrew_text)

    # Save in the following form (similar to CONLLU):
    # # sent_id = sentence_index
    # # text = sentence_hebrew
    # for each word:
    #     word_index word_hebrew word_ornan word_ornan
    # \n
    with codecs.open(file_out, encoding='utf-8', mode='w') as f:
        for sent_idx, sentence in enumerate(hebrew_text):
            f.write(u'# sent_id = {}\n'.format(sent_idx+1))
            f.write(u'# text = {}\n'.format(sentence))
            split_sent = sentence.split(u' ')
            split_sent = list(filter(lambda x: len(x)>0, split_sent))
            for word_idx, word in enumerate(split_sent):
                word_ornan = Hebrew2Ornan(word)
                if IsSkipWord(word_ornan):
                    f.write(u'#')
                f.write(u'{} {} {} {} \n'.format(word_idx+1, word, word_ornan, word_ornan))
            f.write(u'\n')


if __name__=="__main__":
    ConvertRaw('../data/rawAll.txt', '../data/HaaretzOrnan.txt')
