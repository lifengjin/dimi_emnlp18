import sys
import nltk

fn = sys.argv[1]

fn_list = fn.split('.')

ints_fn = fn_list[:-1]
ints_fn.append('ints')
ints_fn = '.'.join(ints_fn)

dic_fn = fn_list[:-1]
dic_fn.append('dict')
dic_fn = '.'.join(dic_fn)
with open(fn, encoding='utf8') as of, open(ints_fn, 'w', encoding='utf8') as intsf, open(dic_fn,
                                                                                      'w',
                                                                                         encoding='utf8'
                                                                                         ) as dicf:
    word_dict = {}
    for line in of:
        try:
            tline = line.strip().lower()
            t = nltk.Tree.fromstring(tline)
            words = t.leaves()

        except:
            line = line.strip().lower()
            words = line.split(' ')
        int_sent = []
        for word in words:
            if word in word_dict:
                int_sent.append(word_dict[word])
            else:
                word_dict[word] = len(word_dict)
                int_sent.append(word_dict[word])
        print(' '.join([str(x) for x in int_sent]), file=intsf)
    for word in word_dict:
        print(word, word_dict[word], file=dicf)