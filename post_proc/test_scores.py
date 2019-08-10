import codecs
from post_processing import romanize


def check_dgeshim(post_processing_manual, post_processing_aotomatic):

    dgeshim = {"h": "", "b": "v", "k": "h", "p": "f", "f": "p", "v": "b"}
    word_aotomatic = list(post_processing_aotomatic)
    word_manual = list(post_processing_manual)
    idx = 0
    let = None
    try:
        for i, letter in enumerate(word_aotomatic):
            if word_aotomatic[i] != word_manual[idx] and letter in dgeshim:
                word_aotomatic[i] = dgeshim.get(letter)
                let = letter
                if letter == "h":
                    idx -= 1
            idx += 1
        if "".join(word_aotomatic) == post_processing_manual:
            if let == "p" or let == "v":
                return "PandB"
            return "dagesh"
    except:#sometimes the number of characters is not equal
        return

def test_scores(file_in, file_out):
    not_equel = []
    dgeshim_problems = []
    counter_dgeshim_problems = 0
    PandB_problems = []
    counter_PandB_problems = 0
    counter_true = 0
    counter_false = 0
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
            post_proc = romanize(line[3])
            if post_proc != line[4]:
                if check_dgeshim(line[4], post_proc) == "dagesh":    #return: true if the problem is only dgeshim else return false
                    dgeshim_problems.append([sent_id, line[0], line[3], line[4], post_proc])
                    counter_dgeshim_problems += 1
                elif check_dgeshim(line[4], post_proc) == "PandB":
                    PandB_problems.append([sent_id, line[0], line[3], line[4], post_proc])
                    counter_PandB_problems += 1
                else:
                    not_equel.append([sent_id, line[0], line[3], line[4], post_proc])
                    counter_false += 1
            else:
                counter_true += 1

    with codecs.open(file_out, encoding='utf-8', mode='w') as f:
        f.write(u'counter_false = {}\n'.format(counter_false))
        f.write(u'counter_true = {}\n'.format(counter_true))
        for i, error in enumerate(not_equel):
            f.write('{}  k: sent_id = {} word = {} {} {}\n'.format(i, error[0], error[1], error[2], error[3]))
            f.write('{}  p: sent_id = {} word = {} {} {}\n'.format(i, error[0], error[1], error[2], error[4]))
        f.write(u'\n\ncounter_PandB_problems = {}\n'.format(counter_PandB_problems))
        for i, error in enumerate(PandB_problems):
            f.write('{}  k: sent_id = {} word = {} {} {}\n'.format(i, error[0], error[1], error[2], error[3]))
            f.write('{}  p: sent_id = {} word = {} {} {}\n'.format(i, error[0], error[1], error[2], error[4]))
        f.write(u'\n\ncounter_dgeshim_problems = {}\n'.format(counter_dgeshim_problems))
        for i, error in enumerate(dgeshim_problems):
            f.write('{}  k: sent_id = {} word = {} {} {}\n'.format(i, error[0], error[1], error[2], error[3]))
            f.write('{}  p: sent_id = {} word = {} {} {}\n'.format(i, error[0], error[1], error[2], error[4]))



if __name__ == '__main__':
    #print(check_dgeshim("se-ki-rim", "she-ki-rim"))
    test_scores('../data/HaaretzOrnan_annotated.txt', 'test_scores.txt')