import codecs
from post_processing import post_processing

def s_h2sh (list_word):
    for i, letter in enumerate(list_word):
        if i > 0 and letter == "h" and list_word[i - 1] == "s":
            list_word[i - 1] = list_word[i - 1] + list_word[i]
            del list_word[i]
    return (list_word)

def statistic (file_in):
    statistic_of_dgeshim = {"sh": [0, 0], "b": [0, 0], "k": [0, 0], "p": [0, 0]}
    with codecs.open(file_in, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if len(line) == 0 or line.startswith(u'#'):
                continue
            line = line.split()
            word_in = s_h2sh(list(post_processing(line[3])))
            word_out = s_h2sh(list(line[4]))
            for i, letter in enumerate(word_in):
                if letter in statistic_of_dgeshim and (i > 0 or letter == "sh"):
                    statistic_of_dgeshim[letter][0] += 1
                    if word_out[i] == letter:
                        statistic_of_dgeshim[letter][1] += 1
    return(statistic_of_dgeshim)




if __name__ == '__main__':
   print(statistic('../data/HaaretzOrnan_annotated.txt'))