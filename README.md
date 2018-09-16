# Depth-bounded grammar Induction Model with Inside-sampling

This repo is for the EMNLP 2018 paper
[**Depth-bounding is effective: Improvements and evaluation of unsupervised
PCFG induction**](http://arxiv.org/abs/1809.03112).
Please cite the paper if you
use this repo in your work:

```
﻿@inproceedings{Jin2018b,
author = {Jin, Lifeng and Doshi-Velez, Finale and Miller, Timothy A and Schuler, William and Schwartz, Lane},
booktitle = {EMNLP},
title = {{Depth-bounding is effective: Improvements and evaluation of unsupervised PCFG induction}},
year = {2018}
}
```

This repo is built partly on [UHHMM](https://github.com/tmills/uhhmm) and
[DB-PCFG](https://github.com/lifengjin/db-pcfg) repos.

## Prerequisits

Required Python3 packages
- bidict
- numba
- numpy
- nltk
- scipy
- pyzmq

## Usage

### Single Machine

Using this repo is very straightforward.
1. You need a file which is tokenized,
delimited by space for words and one sentence per line.
2. Then use `utils/make_ints_file.py` to
process the file into two files, `{*}.ints` and `{*}.dict`.
3. Create a config file following the format below. You can also find a
sample config file in the repo.
4. You can run `python dimi-trainer.py config-file.ini` now.
5. Useful files will be generated in a output directory.

```
[io]
input_file = ./genmodel/ptb_20.dev.linetoks.ints
output_dir = ./outputs/WSJ20
dict_file = ./genmodel/ptb_20.dev.linetoks.dict

[params]
iters = 700
K = 15
init_alpha = .2
cpu_workers = 18
D = 2
batch_per_worker = 200

```

`input_file`: path to a `ints` file
`output_dir`: path to a directory where the output files will be written
`dict_file`: path to a `dict` file
`iters`: total number of iterations to run
`K`: total number of non-terminal categories
`init_alpha`: the symmetric Dirichlet prior parameter
`cpu_workers`: number of processes
`D`: maximum allowed depthf for induced grammars
`batch_per_worker`: number of sentences per batch per worker

### Cluster

It is also easy to use this with a cluster. The steps are essentially the same as the
 single machine usage. The only differences are:

1. You need to set cpu_workers to 0 in your config to tell the master that it
is a master.
2. You need to run `python start_cluster_worker/py --config-path .` for each worker
you want to launch. This may be achieved in a cluster by submitting an array job
and ask for one core for each worker.
3. That's it.

### Continue a run

You use a config file only when you want to start a new run. In order to continue an
old run, you can do `python dimi-trainer.py the/path/to/output_dir`, and
the program will read the models and config from there to continue a run.
The single machine and cluster steps still apply.

## Output

After running for a while, you may see some files in your output directory.
First, the output directory is named like this : `output_dir_D*K*A*_i`, where
`*`s are hyper-parameters set by you, and `i` is the index of the directory to avoid overwriting.
Therefore you can run multiple experiments with the same config file without
worrying about they will overwrite each other.

In your output directory, these useful files may be generated:
- `iter_{k}.linetrees.gz`: `k` is the iteration number. These are the sampled trees
for the corpus you use. You should be able to view them with `zless` or `less`
directly.
- `pcfg_model_{k}.pkl`: The program saves the whole PCFG model for the workers to use,
but only keeps the most recent three model files around, as well as every 100th
model. These are the models that are saved.
- `pcfg_hypparams.txt`: This is a log file for some important runtime statistics.
The commonly used ones include loglikelihood and right-branching score.
- `pcfg_counts_info.txt`: Records the raw counts of each non-terminals categories.
The total occurrences and the terminal only occurrences are recorded.
- `log.txt`: This is a general log file of runtime information with timestamps.

## Other helpful scripts
Please visit the `utils` folder to view its Readme for notes on some convenience
tools this repo supplies.

## Example
You can start an example run by doing `python dimi-trainer.py config/config.ini`
with the repo. It runs the center-embedding synthetic dataset provided in `datasets`.
You should see with a high probability that the run converge at around -1750
loglikelihood with a right-branching score of 0.38. There is a chance you
are not able to get these results on the first try, so please try a couple
more runs.
