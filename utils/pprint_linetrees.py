#!/usr/bin/env python3

import nltk
import gzip
import argparse
import os
import sys


def pprint_linetrees(in_file, out_file):

    print("Reading linetrees from \t" + os.path.abspath(in_file.name), file=sys.stderr)
    print("Writing pretty trees to\t" + os.path.abspath(out_file.name), file=sys.stderr)

    i = 0
    for line in in_file:
        tree = nltk.Tree.fromstring(line)
        print('##############Tree', str(i), file=out_file)
        tree.pretty_print(stream=out_file)
        i += 1




if __name__ == "__main__":

    if len(sys.argv) == 2 and sys.argv[1]=='-':

        pprint_linetrees(sys.stdin, sys.stdout)

    else:

        parser = argparse.ArgumentParser()
        parser.add_argument('--linetrees','-l', type=str, required=True, help='the path to the linetrees ' \
                                                                    'that needs to be '
                                                                    'translated to pprint trees')
        args = parser.parse_args()

        filename = args.linetrees

        if filename.endswith('linetrees.gz'):
            out_filename=filename.replace('.linetrees.gz', '.pplinetrees')
            with gzip.open(filename, 'rt', encoding='utf8') as in_file, open(out_filename, 'w', encoding='utf8') as out_file:
                pprint_linetrees(in_file, out_file)
        
        elif filename.endswith('linetrees'):
            out_filename=filename.replace('.linetrees', '.pplinetrees')
            with open(filename, 'rt', encoding='utf8') as in_file, open(out_filename, 'w', encoding='utf8') as out_file:
                pprint_linetrees(in_file, out_file)
            
        else:
            out_filename=filename+'.pplinetrees'
            with open(filename, 'rt', encoding='utf8') as in_file, open(out_filename, 'w', encoding='utf8') as out_file:
                pprint_linetrees(in_file, out_file)
