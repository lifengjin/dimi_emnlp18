import argparse
import gzip
import multiprocessing
import collections
import os
import random
import re
import nltk
from tqdm import tqdm

'''
python utils/MAP_constituent.py --dirs ./wsj20dev_D2K15/ ./wsj20dev_D2K15_1/ 
./wsj20dev_D2K15_2 ./wsj20dev_D2K15_3 
./wsj20dev_D2K15_4 ./wsj20dev_D2K15_5 
./wsj20dev_D2K15_6 ./wsj20dev_D2K15_7 
./wsj20dev_D2K15_9 --process 15 --output-fn test.txt.linetrees.gz --max-iter 650 --min-iter 550
'''

def read_linetrees_file(fn):
    with gzip.open(fn, 'rt') as fh:
        trees = fh.readlines()
        # for line in fh:
        #     tree = nltk.Tree.fromstring(line)
        #     trees.append(tree)
    return trees

def add_index_to_leaves(tree):
    for leaf_index, leaf_position in enumerate(tree.treepositions(order='leaves')):
        tree[leaf_position] = tree[leaf_position] + '||' + str(leaf_index)

def argmax_bottom_up(dic, constituent_units):
    max_keys = []
    max_val = 0
    max_list_index = []
    bigram_tuples = [ (const, constituent_units[index+1]) for index, const in enumerate(
        constituent_units[:-1])]
    bigrams = [' '.join([const, constituent_units[index+1]]) for index, const in enumerate(
        constituent_units[:-1])]
    for index, key in enumerate(bigrams):
        if dic[key] > max_val:
            max_val = dic[key]
            max_keys = [key]
            max_list_index = [index]
        elif dic[key] == max_val:
            max_keys.append(key)
            max_list_index.append(index)
    if len(max_list_index) == 1:
        max_index = max_list_index[0]
    else:
        max_index = random.choice(max_list_index)
    return bigram_tuples[max_index], max_val

def argmax_top_down_nary(tree_const_list, constituent_units):
    if len(constituent_units) == 1:
        return [tuple(constituent_units), ], ['(' + re.sub('\|\|\d+', '', constituent_units[0]) +
                                              ') :1', ]
    solution = []
    prob_strings = []
    max_keys = []
    max_val = 0
    max_list_index = []
    bigram_tuples = []
    for i in range(1, len(constituent_units)):
        bigram_tuples.append((constituent_units[:i], constituent_units[i:]))
    bigrams = [(' '.join(left), ' '.join(right)) for left, right in bigram_tuples]
    counts = {x: 0 for x in bigrams}
    for tree_index, tree_const in enumerate(tree_const_list):
        for const in bigrams:
            left = const[0]
            right = const[1]
            if left in tree_const and right in tree_const:
                counts[const] += 1
    for bigram in counts:
        if counts[bigram] > max_val:
            max_val = counts[bigram]
            max_keys = [bigram]
            max_list_index = [bigrams.index(bigram)]
        elif counts[bigram] == max_val:
            max_keys.append(bigram)
            max_list_index.append(bigrams.index(bigram))
    if len(max_keys) == 1:
        max_index = max_list_index[0]
    else:
        max_index = random.choice(max_list_index)
    total_counts = sum(counts.values())
    vals = [counts[bigram] for bigram in bigrams]
    sorted_vals = sorted(vals, reverse=True)
    # vals.sort(reverse=True)
    # assert vals[0] == max_val
    probs = [x / total_counts for x in vals]
    prob_string = '(' + ') , ('.join([re.sub('\|\|\d+', '', x) for x in bigrams[max_index]]) + ') ' \
                                       ':' + ' '.join(['{:.4f}'.format(x) for x in probs])
    sorted_probs = [ x / total_counts for x in sorted_vals]
    if  (len(constituent_units) == 4 or len(constituent_units) == 3) and sorted_probs[0] - \
            sorted_probs[1] <= 0.3 and flattening:
        solution.append(tuple([x,] for x in constituent_units))
        prob_strings.append(prob_string)
        # return solution, prob_strings
    else:
        solution.append(bigram_tuples[max_index])
        prob_strings.append(prob_string)
    for const in solution[-1]:
        part_solution, prob_string = argmax_top_down_nary(tree_const_list, const)
        solution.extend(part_solution)
        prob_strings.extend(prob_string)
    return solution, prob_strings

def argmax_top_down(tree_const_list, constituent_units):
    if len(constituent_units) == 1:
        return [tuple(constituent_units),],[ '(' + re.sub('\|\|\d+', '', constituent_units[0]) +
                                                          ') :1',]
    solution = []
    prob_strings = []
    max_keys = []
    max_val = 0
    max_list_index = []
    bigram_tuples = [ ]
    for i in range(1, len(constituent_units)):
        bigram_tuples.append((constituent_units[:i], constituent_units[i:]))
    bigrams = [(' '.join(left), ' '.join(right)) for left, right in bigram_tuples]
    counts = {x : 0 for x in bigrams}
    for tree_index, tree_const in enumerate(tree_const_list):
        for const in bigrams:
            left = const[0]
            right = const[1]
            if left in tree_const and right in tree_const:
                counts[const] += 1
    for bigram in counts:
        if counts[bigram] > max_val:
            max_val = counts[bigram]
            max_keys = [bigram]
            max_list_index = [bigrams.index(bigram)]
        elif counts[bigram] == max_val:
            max_keys.append(bigram)
            max_list_index.append(bigrams.index(bigram))
    if len(max_keys) == 1:
        max_index = max_list_index[0]
    else:
        max_index = random.choice(max_list_index)
    total_counts = sum(counts.values())
    vals = [ counts[bigram] for bigram in bigrams ]
    # vals.sort(reverse=True)
    # assert vals[0] == max_val
    probs = [x / total_counts for x in vals]
    prob_string = '(' + ') , ('.join([re.sub('\|\|\d+', '', x) for x in bigrams[max_index]]) + ') ' \
                              ':' + ' '.join(['{:.4f}'.format(x) for x in probs])
    solution.append(bigram_tuples[max_index])
    prob_strings.append(prob_string)
    for const in bigram_tuples[max_index]:
        part_solution, prob_string = argmax_top_down(tree_const_list, const)
        solution.extend(part_solution)
        prob_strings.extend(prob_string)
    return solution, prob_strings

def process_single_tree_bottom_up(tree_list):
    tree_list = [nltk.Tree.fromstring(t) for t in tree_list]
    span_counter = collections.Counter()
    for tree in tree_list:
        spans = []
        add_index_to_leaves(tree)
        for subtree in tree.subtrees():
            spans.append(' '.join(tuple(subtree.leaves())))
        span_counter.update(spans)
    sent = tree_list[0].leaves()
    solution = []
    while len(sent) > 1:
        (max_left, max_right), max_count = argmax_bottom_up(span_counter, sent)
        solution.append((max_left, max_right))
        left_index = sent.index(max_left)
        right_index = sent.index(max_right)
        sent[left_index] = ' '.join((max_left, max_right))
        del sent[right_index]
    solution += [(x,) for x in tree_list[0].leaves()]
    # solution = clean_solution(solution)
    sent = tree_list[0].leaves()
    return solution_to_tree(sent, solution)

def process_single_tree_top_down(tree_list):
    tree_list = [nltk.Tree.fromstring(t) for t in tree_list]
    all_tree_spans = []
    for tree in tree_list:
        this_spans = []
        add_index_to_leaves(tree)
        for subtree in tree.subtrees():
            this_spans.append(' '.join(tuple(subtree.leaves())))
        all_tree_spans.append(this_spans)
    sent = tree_list[0].leaves()

    # solution, prob_strings = argmax_top_down(all_tree_spans, sent)
    solution, prob_strings = argmax_top_down_nary(all_tree_spans, sent)
    # print(solution)
    for index, const in enumerate(solution):
        if len(const) > 1:
            children = []
            for child in const:
                child = ' '.join(child)
                children.append(child)
            solution[index] = tuple(children)

    sent = tree_list[0].leaves()
    return solution_to_tree(sent, solution), prob_strings

def test():
    t1 = "(X (X (X (X a) (X b)) (X c)) (X d))"
    t2 = "(X (X (X a) (X b)) (X (X c) (X d)))"
    t3 = "(X (X a) (X (X b) (X (X c) (X d))))"
    t4 = "(X (X a) (X (X b) (X (X c) (X d))))"

    ts = [t1, t2, t3, t4]
    # sent = nltk.Tree.fromstring(t1)
    # add_index_to_leaves(sent)
    # sent = sent.leaves()
    # print(sent)
    solution = process_single_tree_top_down(ts)
    print(solution)
    # print(solution_to_tree(sent, solution))
    solution = process_single_tree_bottom_up(ts)
    print(solution)
    # print(solution_to_tree(sent, solution))

def solution_to_tree(string, solution):
    if isinstance(string, str):
        string = string.split(' ')
    str_len = len(string)
    for consts in solution:
        if len(consts) > 1:
            const_span_len = sum([len(x.split(' ')) for x in consts])
            first_word = consts[0].split(' ')[0]
        else:
            const_span_len = 1
            first_word = consts[0]
        if const_span_len == str_len and string[0] == first_word:
            if len(consts) > 1:
                children = []
                for child in consts:
                    child = solution_to_tree(child, solution)
                    children.append(child)
            else:
                return nltk.Tree('X', children=[re.sub('\|\|\d+', '', first_word)])
            this_tree = nltk.Tree('X', children=children)
            return this_tree

def wrap_file_func(i, arg):
    return i, read_linetrees_file(arg)

def wrap_bottom_up_func(i, arg):
    return i, process_single_tree_bottom_up(arg)

def wrap_top_down_func(i, arg):
    return i, process_single_tree_top_down(arg)

def update(i_ans):
    i = i_ans[0]
    ans = i_ans[1]
    best_trees[i] = ans
    # print(ans[0])
    pbar.update()

def file_update(i_ans):
    i = i_ans[0]
    ans = i_ans[1]
    trees[i] = ans
    pbar.update()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', '-d', nargs='+', help='the directories that contain the linetree '
                                                     'samples')
    parser.add_argument('--mode', '-m', choices=['tree', 'const'], help='the MAP mode: on trees or on '
                                                                      'const[ituents]')
    parser.add_argument('--processes', '-p', type=int, help='number of parallel processes')
    parser.add_argument('--order', '-o', default='top-down', choices=['bottom-up', 'top-down'], help='which order is '
                                                                                 'the optimal '
                                                                                 'tree generated')
    parser.add_argument('--test', '-t', action='store_true', default=False)
    parser.add_argument('--output-fn', '-of', type=str, help='name of the output file')
    parser.add_argument('--max-iter', type=int, default=10000, help='the maximum iteration '
                                                                  'linetrees ' \
                                                             'file used')
    parser.add_argument('--min-iter', type=int, default=-1, help='the minimum iteration linetrees '
                                                               'file ' \
                                                         'used')
    parser.add_argument('--flattening', default=False, action='store_true', help='turn on '
                                                                                 'posterior '
                                                                                 'flattening')
    args = parser.parse_args()
    if args.test:
        test()
        exit(0)
    flattening = args.flattening

    dirs = args.dirs

    fns = []

    best_trees = []
    print('Reading in the directory info.')
    for directory in dirs:
        for f in os.listdir(directory):
            if re.match("iter_[\d]+\.linetrees\.gz", f):
                iter_num = re.search('(?<=er_)\d+', f).group(0)
                if int(iter_num) < args.max_iter and int(iter_num) > args.min_iter:
                    fns.append(os.path.join(directory,f))

    print('Processing {} individual files.'.format(len(fns)))
    with multiprocessing.Pool(args.processes) as pool:

        pbar = tqdm(total=len(fns))
        trees = [None] * len(fns)
        for i in range(len(fns)):
            pool.apply_async(wrap_file_func, args=(i, fns[i]),
                                   callback=file_update)
        pool.close()
        pool.join()
        pbar.close()

    with multiprocessing.Pool(args.processes) as pool:
        print('Zipping the trees together.')
        trees_for_single_sent_list = list(zip(*trees))
        print(len(trees_for_single_sent_list), 'number of trees to be processed.')

        print("Processing individual trees.")
        pbar = tqdm(total=len(trees_for_single_sent_list))
        best_trees = [None] * len(trees_for_single_sent_list)
        for i in range(len(trees_for_single_sent_list)):
            if args.order == 'bottom-up':
                pool.apply_async(wrap_bottom_up_func, args=(i, trees_for_single_sent_list[i]),
                                   callback=update)
            elif args.order == 'top-down':
                pool.apply_async(wrap_top_down_func, args=(i, trees_for_single_sent_list[i]),
                                 callback=update)
        pool.close()
        pool.join()
        pbar.close()

    print("Writing out results.")
    with gzip.open(args.output_fn, 'wt') as ofh:
        if args.order == 'top-down':
            probfh = gzip.open(args.output_fn+'.probs', 'wt')
        for tree in best_trees:
            # print(tree)
            if args.order == 'top-down':
                some_tree, probs = tree
                string = some_tree.pformat(margin=100000)
                print('SENT:', string, file=probfh)
                for item in probs:
                    # print(item)
                    assert isinstance(item, str)

                    print(item, file=probfh)
            else:
                some_tree = tree

                string = some_tree.pformat(margin=100000)
            print(string, file=ofh)
        probfh.close()