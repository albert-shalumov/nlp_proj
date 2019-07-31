import codecs


def verify_score():
    scores = (u'a',u'o',u'i',u'e',u'u',u'*')
    with codecs.open('../data/HaaretzOrnan_annotated.txt', encoding='utf-8') as f:
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
            if line[3].count(u'-') != line[4].count(u'-'):
                print('the - is not equal in sent_id:', sent_id, 'word:', line[0])
            word = line[3].replace(u'-', u'')
            for i in word[1::2] + word[len(word)-1]:
                if i not in scores:
                    print('there is an error in sent_id:', sent_id, 'word:', line[0])
                    break



if __name__ == '__main__':
   verify_score()