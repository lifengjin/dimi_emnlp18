from nltk import tree
import argparse
from collections import defaultdict
import re
from typing import List

parser = argparse.ArgumentParser()

parser.add_argument('--linetrees', type=str, required=True)#, required=True)
parser.add_argument('--factor', type=str, default='right')
args = parser.parse_args()

# depth starts with 0, and the top discourse depth is -1
def get_max_depth(tree : tree.Tree, factor : str ='right') -> int:
    tree.collapse_unary()
    max_depth = 0

    tree.chomsky_normal_form(factor=factor)

    leaf_positions = tree.treepositions('leaves')

    for leaf_p in leaf_positions:
        p_str = '0'+''.join([str(x) for x in leaf_p[:-1]])
        turns = re.findall('0[1-9]', p_str)
        this_depth = len(turns)
        if this_depth > max_depth:
            max_depth = this_depth
    if max_depth == 0 and len(leaf_positions) != 1:
        print(leaf_positions)
        print(tree)
        raise Exception
    # if max_depth[0] != max_depth[1]:
    #     print(tree)
    #     tree.un_chomsky_normal_form()
    #     print(tree)
    #     tree.chomsky_normal_form(factors[0])
    #     print(tree)
    #
    #     raise Exception
    return max_depth



tree_depths = defaultdict(int)
if args.linetrees.endswith('gz'):
    import gzip
    alltrees = gzip.open(args.linetrees, 'rt', encoding='utf8')
else:
    alltrees = open(args.linetrees, encoding='utf8')
for line in alltrees:
    linetree = tree.Tree.fromstring(line.strip())
    depth = get_max_depth(linetree, args.factor)
    tree_depths[depth] += 1

total = sum(tree_depths.values())
acc = 0
for d in range(len(tree_depths)):
    acc += tree_depths[d]
    val = tree_depths[d]
    print('depth {}: {} {}, {:.2f}%, accu {:.2f}%'.format(d, args.factor, tree_depths[d],
                                                          val / total * 100
                                                                          , acc / total * 100
                                                                          ))
