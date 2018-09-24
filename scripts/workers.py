#!/usr/bin/env python3

import logging
from multiprocessing import Process, get_start_method
import os, os.path
import signal
import subprocess
import sys
import time
from .PyzmqWorker import *

def start_cluster_workers(work_distributer, cluster_cmd, maxLen, gpu, K, D, batch_size):
    logging.debug("Cluster command is %s" % cluster_cmd)

    cmd_str = 'python3 %s/scripts/workers.py -host %s -jobs-port %d -results-port %d ' \
              '-models-port %d -max-len %d -gpu %d -K %d -D %d -batch-size %d' % (os.getcwd(),
                                                                  work_distributer.host,
                                                                   work_distributer.jobs_port,
                                                                   work_distributer.results_port,
                                                                   work_distributer.models_port,
                                                                   maxLen, int(gpu), K, D, batch_size)
    submit_cmd = [ cmd_arg.replace("%c", cmd_str) for cmd_arg in cluster_cmd.split()]
    logging.info("Making cluster submit call with the following command: %s" % str(submit_cmd))
    subprocess.call(submit_cmd)

def start_local_workers_with_distributer(work_distributer, maxLen, cpu_workers, gpu_workers=0,
                                         gpu=False, batch_size=1, K=None, D=None):
    logging.info("Starting workers with maxLen=%d and num_cpu_workers=%d and num_gpu_workers=%d" % (maxLen, cpu_workers, gpu_workers) )
    return start_local_workers(host=work_distributer.host, jobs_port=work_distributer.jobs_port,
                               results_port=work_distributer.results_port,
                               models_port=work_distributer.models_port, maxLen=maxLen,
                               cpu_workers=cpu_workers, gpu_workers=gpu_workers, gpu=gpu,
                               batch_size=batch_size, K=K, D=D)

def worker_run_proxy(worker_list, i, config_list=None):
    worker = worker_list[i]
    if config_list is not None:
        config = config_list[i]
        host, jobs_port, results_port, models_port, maxLen, K, D, tid, gpu , batch_size, level = config
        worker = PyzmqWorker(host, jobs_port, results_port, models_port, maxLen, K=K, D=D,
                                     tid=tid, gpu=gpu, batch_size=batch_size, level=level)
    worker.run()

def start_local_workers(host=None, jobs_port=None, results_port=None, models_port=None,
                        maxLen=None, K=None, D=None, cpu_workers=None,
                        gpu_workers=0, gpu=False, batch_size=1):
    logging.info("Starting %d cpu workers and %d gpu workers at host %s with jobs_port=%d, results_port=%d, models_port=%d, maxLen=%d" % (cpu_workers, gpu_workers, host, jobs_port, results_port, models_port, maxLen) )
    processes = []
    workers = []
    worker_configs = []
    for i in range(0, cpu_workers+gpu_workers):
        if i >= gpu_workers:
            gpu = False
        else:
            gpu = True
        # logging.info("distributer starts with D {} and K {}".format(D, K))
        fs = PyzmqWorker(host, jobs_port, results_port, models_port, maxLen, K=K, D=D,
                                     tid=i, gpu=gpu, batch_size=batch_size, level=logging.getLogger().getEffectiveLevel())
        worker_configs.append((host, jobs_port, results_port, models_port, maxLen, K, D,
                                     i, gpu, batch_size, logging.getLogger().getEffectiveLevel()))
        workers.append(fs)

    for i in range(0, cpu_workers+gpu_workers):
        if get_start_method() == 'spawn':
            p = Process(target=worker_run_proxy, args=(workers,i, worker_configs))
        else:
            p = Process(target=worker_run_proxy, args=(workers,i))
        p.daemon = True
        processes.append(p)
        p.start()

    return processes

