#!/usr/bin/env python3

import logging
import os.path
import pickle
import socket
import time
import zmq
from queue import Queue
from .PyzmqMessage import SentenceJob, CompileJob, PyzmqJob, SentenceRequest, RowRequest, get_file_signature, resource_current, ModelLocation
from threading import Thread, Lock
import sys

class ResetSignal():
    def __init__(self):
        self.reset = True

class VerboseLock():
    def __init__(self, name):
        self.name = name
        self._lock = Lock()

    def acquire(self):
        logging.debug("Acquiring %s lock" % self.name)
        t0 = time.time()
        self._lock.acquire()
        t1 = time.time()
        if t1-t0 > 1:
            logging.warning("Acquiring %s lock took %d seconds" % (self.name, t1-t0))

    def release(self):
        logging.debug("Releasing %s lock" % self.name)
        self._lock.release()

class Ventilator(Thread):
    def __init__(self, host, sync_port, sent_list):
        Thread.__init__(self)
        self.host = host
        self.sent_list = sent_list
        logging.debug("Job distributer attempting to bind to PUSH socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://"+self.host)

        logging.debug("Ventilator successfully bound to PUSH socket.")

        self.sync_socket = context.socket(zmq.REQ)
        self.sync_socket.connect("tcp://%s:%s" % (self.host, sync_port))
        logging.debug("Ventilator connected to sync socket.")

        self.job_queue = Queue()

    def run(self):
        while True:
            ## Wait for signal to start:
            logging.debug("Ventilator waiting for permission to start")
            self.sync_socket.send(b'0')
            sync = self.sync_socket.recv_pyobj()

            if sync == b'0':
                break
            else:
                current_resource_sig = sync

            logging.debug("Ventilator received model signature sync signal")
            while not self.job_queue.empty():
                job_request = self.socket.recv_pyobj()
                worker_resource_sig = job_request.resource_sig

                if not resource_current(current_resource_sig, worker_resource_sig):
                    ## Send them a quit message:
                    quit_job = PyzmqJob(PyzmqJob.QUIT, None)
                    if job_request.request_size > 1:
                        quit_job = [quit_job]

                    self.socket.send_pyobj(quit_job)
                    continue

                jobs = []

                while not self.job_queue.empty() and len(jobs) < job_request.request_size:
                    job = self.job_queue.get()
                    jobs.append(job)
                    self.job_queue.task_done()

                logging.log(logging.DEBUG-1, "Ventilator pushing job %d" % job.resource.index)

                if job_request.request_size > 1:
                    self.socket.send_pyobj(jobs)
                else:
                    self.socket.send_pyobj(jobs[0])

            logging.debug("Ventilator iteration finishing")

        logging.debug("Ventilator thread finishing")
        self.socket.close()
        self.sync_socket.close()
        logging.debug("All ventilator sockets closed.")

    def addJob(self, job):
        self.job_queue.put(job)

class Sink(Thread):
    def __init__(self, host, sync_port, num_sents):
        Thread.__init__(self)
        self.host = host
        self.num_sents = num_sents
        self.outputs = list()
        logging.debug("Parse accumulator attempting to bind to PULL socket...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)

        self.port = self.socket.bind_to_random_port("tcp://"+self.host)

        logging.debug("Parse accumulator successfully bound to PULL socket.")

        self.sync_socket = context.socket(zmq.REQ)
        self.sync_socket.connect("tcp://%s:%s" % (self.host, sync_port))
        logging.debug("Sink connected to sync socket.")

        self.work_lock = VerboseLock("Sink")
        self.processing = False
        self.batch_size = self.num_sents

        self.model_rows = {}

    def run(self):

        while True:
            logging.debug("Sink waiting for permission to start...")
            self.sync_socket.send(b'0')
            sync = self.sync_socket.recv_pyobj()
            if sync == b'0':
                break

            model_sig = sync

            logging.debug("Sink received model signature sync signal")

            num_done = 0
            self.outputs = list()
            self.model_rows = dict()

            while num_done < self.batch_size:
                try:
                    job_outcome = self.socket.recv_pyobj()
                    num_done += 1
                    if job_outcome.job_type == PyzmqJob.SENTENCE:
                        parse = job_outcome.result
                        parse.success = job_outcome.success
                        if not parse.success:
                            # logging.info(logging.DEBUG-1, "Sink received parse %d with result: "
                            #                                "%s" % (parse.index, list(map(lambda x: x.str(), parse.state_list))))
                        # else:
                            logging.warning("Sink received parse %d with parse failure." % (
                                parse.index))

                        self.outputs.append(parse)
                    else:
                        logging.warning("Received a job with an unknown job type!")
                        raise Exception
                except Exception as e:
                    logging.error("Sink caught an exception waiting for parse: %s" % (e) )
                    break

            logging.debug("Sink finished processing this batch of sentences")

            self.setProcessing(False)

        logging.debug("Sink thread finishing")
        self.socket.close()
        self.sync_socket.close()
        logging.debug("All sink sockets closed.")
        self.setProcessing(False)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def setProcessing(self, val):
        self.work_lock.acquire()
        self.processing = val
        self.work_lock.release()

    def getProcessing(self):
        val = self.processing
        return val

    def get_parses(self):
        # print(len(self.outputs))
        # for i in range(len(self.outputs)):
        #     print('sink', i, self.outputs[i].state_list)
        return self.outputs

class ModelDistributer(Thread):
    def __init__(self, host, sync_port, working_dir):
        Thread.__init__(self)
        self.host = host
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.quit_socket = context.socket(zmq.REQ)

        ## disconnect every TO ms to check for quit flag.
        self.port = self.socket.bind_to_random_port("tcp://"+self.host)
        self.quit_socket.connect("tcp://%s:%s" % (self.host, self.port))

        logging.debug("Model server successfully bound to REP socket")
        self.working_dir = os.path.abspath(working_dir)
        self.model_sig = None
        self.model_lock = VerboseLock("Model")
        self.quit = False
        self.model_loc = ModelLocation(self.host, os.path.join(self.working_dir, '/models.bin'))
        
    ## All this method does is wait for requests for the model and send them,
    ## with a quick check to make sure that the model isn't currently being written
    def run(self):

        
        ## Wait until we're actually given a model to start sending them out...
        while self.model_sig is None:
            time.sleep(1)

        while True:
            ## TODO -- put a timeout in this recv() of a second so that it checks the
            ## quit flag regularly
            try:
                sync = self.socket.recv()
                if sync == b'-1':
                    logging.info("Model server received quit signal")
                    break
                logging.log(logging.DEBUG, 'Sending worker a model in response to signal %s' % str(sync))
                self.socket.send_pyobj(self.model_loc)
                ## Don't need to do anything -- this happens when there is a timeout,
                ## and just need to check the quit value regularly. Don't know when
                ## to quit otherwise because we can't be sure of how many workers there
                ## are and that they've all quit.
            except:
                logging.debug("Model server exception %s... checking quit flag." % str(e))
                raise

        self.socket.close()

    def reset_models(self, fn='models.bin'):
        self.model_loc.file_path = os.path.join(self.working_dir, fn)
        self.model_sig = get_file_signature(self.model_loc.file_path)

    def send_quit(self):
        self.quit_socket.send(b'-1')

class WorkDistributerServer():

    def __init__(self, sent_list, working_dir):

        ## Set up job distribution servers:
        self.host = get_local_ip()
        self.sent_list = sent_list

        context = zmq.Context()

        self.sync_socket = context.socket(zmq.REP)
        sync_port = self.sync_socket.bind_to_random_port("tcp://"+self.host)

        self.vent = Ventilator(self.host, sync_port, sent_list)
        self.vent.daemon = True
        self.sink = Sink(self.host, sync_port, len(sent_list))
        self.sink.daemon = True
        self.model_server = ModelDistributer(self.host, sync_port, working_dir)
        self.model_server.daemon = True

        self.jobs_port = self.vent.port
        self.results_port = self.sink.port
        self.models_port = self.model_server.port

        self.sink.start()
        self.vent.start()

        self.sink_socket = context.socket(zmq.PUSH)
        self.sink_socket.connect("tcp://%s:%s" % (self.sink.host, self.sink.port))

        self.models = None
        self.model_server.start()

    def submitSentenceJobs(self, start=-1, end=-1, sent_index_list=None):
        ind = 0
        num_done = 0
        self.model_server.reset_models()
        model_sig = self.model_server.model_sig
        # print(start, end, 'submit')
        if sent_index_list is not None:
            for i, sent in enumerate(sent_index_list):
                self.vent.addJob(PyzmqJob(PyzmqJob.SENTENCE, SentenceJob(i, self.sent_list[sent])
                                          ) )
            self.sink.setBatchSize(len(sent_index_list))
        elif start >= 0 and end >= 0:
            for i in range(start, end):
                self.vent.addJob(PyzmqJob(PyzmqJob.SENTENCE, SentenceJob(i, self.sent_list[i]) ) )

            self.sink.setBatchSize(end-start)

        self.sink.setProcessing(True)

        ## Wait a bit for sink to process signal and set processing to true for the first time
        time.sleep(3)

        self.startProcessing(model_sig)
        while self.sink.getProcessing():
            time.sleep(0.05)

    def submitBuildModelJobs(self, num_rows, full_pi=False):
        self.model_server.reset_models('raw_models.bin')
        for i in range(0, num_rows):
            compile_job = CompileJob(i, full_pi)
            job = PyzmqJob(PyzmqJob.COMPILE, compile_job)
            self.vent.addJob(job)

        self.sink.setBatchSize(num_rows)
        self.sink.setProcessing(True)

        self.startProcessing(self.model_server.model_sig)

    def startProcessing(self, model_sig):
        self.sync_socket.recv()
        self.sync_socket.send_pyobj(model_sig)
        self.sync_socket.recv()
        self.sync_socket.send_pyobj(model_sig)

    def get_model_row(self, index):
        return self.sink.getModelRow(index)

    def stop(self):
        ## Send two stop signals to sink and ventilator
        logging.debug('Worker Server is stopping.')
        # self.sync_socket.recv()
        # self.sync_socket.send_pyobj(b'0')
        # self.sync_socket.recv()
        # self.sync_socket.send_pyobj(b'0')

        # self.sync_socket.close()
        # self.sink_socket.close()

        # logging.debug('Sending quit.')
        # self.model_server.send_quit()

        logging.info("Waiting for sink thread to join...")
        self.sink.join(1)
        logging.info("Waiting for ventilator thread to join...")
        self.vent.join(1)
        logging.info("Waiting for model server thread to join...")
        self.model_server.join(1)
        logging.info("All threads joined and exiting work distributer server.")

    def get_parses(self):
        return self.sink.get_parses()

## This function was required because of some funkiness on ubuntu systems where reverse dns lookup was returning a loopback ip
## This will try the easy way and if it returns something with 127. will make an outside connectino to known DNS (8.8.8.8) and
## grab the external IP from that socket.
## Solution from here: http://stackoverflow.com/a/1267524
def get_local_ip():
    try:
        return [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.0.1")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
    except:
        return "127.0.0.1"
