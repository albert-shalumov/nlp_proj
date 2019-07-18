import codecs


def Hebrew2Ornan(w):
    return ""

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

    ornan_text = list()
    for word in hebrew_text:
        ornan_text.append(Hebrew2Ornan(word))

    # Save in the following form (similar to CONLLU):
    # # sent_id = sentence_index
    # # text = sentence_hebrew
    # for each word:
    #     word_index word_hebrew word_ornan word_ornan
    # \n
    with codecs.open(file_out, encoding='utf-8', mode='w') as f:
        pass

if __name__=="__main__":
    ConvertRaw('../data/rawAll.txt', '../data/HaaretzOrnan.txt')
