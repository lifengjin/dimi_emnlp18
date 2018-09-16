#!/usr/bin/env python3

import logging
from multiprocessing import Process
import multiprocessing
import os, os.path
import signal
import subprocess
import sys
import time
import scripts.PyzmqWorker
import argparse

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

def start_local_workers(host=None, jobs_port=None, results_port=None, models_port=None,
                        maxLen=None, K=None, D=None, cpu_workers=None,
                        gpu_workers=0, gpu=False, batch_size=1):
    logging.info("Starting %d cpu workers and %d gpu workers at host %s with jobs_port=%d, results_port=%d, models_port=%d, maxLen=%d" % (cpu_workers, gpu_workers, host, jobs_port, results_port, models_port, maxLen) )
    processes = []

    for i in range(0, cpu_workers+gpu_workers):
        if i >= gpu_workers:
            gpu = False
        else:
            gpu = True
        # logging.info("distributer starts with D {} and K {}".format(D, K))
        fs = PyzmqWorker.PyzmqWorker(host, jobs_port, results_port, models_port, maxLen, K=K, D=D,
                                     tid=i, gpu=gpu, batch_size=batch_size, level=logging.getLogger().getEffectiveLevel())
        # signal.signal(signal.SIGTERM, fs.handle_sigterm)
        # signal.signal(signal.SIGINT, fs.handle_sigint)
        # signal.signal(signal.SIGALRM, fs.handle_sigalarm)
        # if gpu:
        #     ## Workers 0-(gpu_workers-1) are the gpu workers -- assign them to
        #     ## cuda devices 0-(gpu_workers-1)
        #     gpu_num = i % 8
        # p = Process(target=fs.run, kwargs={'gpu_num':gpu_num})
        p = Process(target=fs.run)
        p.daemon = True
        processes.append(p)
        p.start()

    return processes

def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if args.config_path:
        config_file = args.config_path + "/masterConfig.txt"
        while True:
            if os.path.isfile(config_file):
                configs = open(config_file).readlines()
                if len(configs)==2 and 'OK' in configs[1]:
                    logging.info('OSC setup acquired. Starting a worker with ' + config_file)
                    args = eval(configs[0].strip())
                    break
            else:
                time.sleep(3)

    num_workers = args.get('cpu_workers', 0) + args.get('gpu_workers', 0)
    if num_workers == 0:
        if args['gpu']:
            args['gpu_workers'] = 1
            args['cpu_workers'] = 0
        else:
            args['cpu_workers'] = 1
            args['gpu_workers'] = 0
        num_workers = 1

    processes = start_local_workers(host=args['host'], jobs_port=args['jobs_port'],
                                    results_port=args['results_port'],
                                    models_port=args['models_port'], maxLen=args['max_len'],
                                    cpu_workers=args['cpu_workers'], gpu_workers=args[
                                    'gpu_workers'], gpu=args['gpu'],
                                    batch_size=args['batch_size'], K=args['K'], D=args['D'])

    for i in range(0, num_workers):
        processes[i].join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-host', type=str)
    parser.add_argument('-jobs-port', type=int)
    parser.add_argument('-results-port', type=int)
    parser.add_argument('-models-port', type=int)
    parser.add_argument('-max-len', type=int)
    parser.add_argument('-cpu-workers', type=int, default=0)
    parser.add_argument('-gpu-workers', type=int, default=0)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-K', type=int)
    parser.add_argument('-D', type=int)
    parser.add_argument('-batch-size', type=int)
    parser.add_argument('-config-path', type=str)
    args = parser.parse_args()
    main(args)
