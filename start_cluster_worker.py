
import os.path
from scripts.workers import *
import argparse

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
