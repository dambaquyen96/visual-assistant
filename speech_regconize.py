import argparse
from ws4py.client.threadedclient import WebSocketClient
import threading
import urllib
import queue
import multiprocessing
import json
import os
import array

class SpeechRecognize(WebSocketClient):

    def __init__(self, url, buffers=None, protocols=None, 
                extensions=None, heartbeat_freq=None, byterate=32000, 
                output_file=None, one_sentence=False, logging=False):
        super(SpeechRecognize, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.buffers = multiprocessing.Queue()
        self.final_hyps = []
        self.byterate = byterate
        self.final_hyp_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.output_file = output_file
        self.one_sentence = one_sentence
        self.logging = logging
        self.alive = True

        self.connect()

    def opened(self):
        def write_audio(q):
            while True:
                block = q.get()
                self.send(block, binary=True)
            f.close()
            self.close()

        p2 = multiprocessing.Process(target=write_audio, args=((self.buffers,)))
        p2.start()

    def push_audio(self, block):
        self.buffers.put(block)

    def received_message(self, m):
        if m.is_text:
            recvStr = m.data.decode("utf-8")
            msg = json.loads(str(recvStr))
            if msg['status'] == 0:
                text = msg['result']['hypotheses'][0]['transcript']
                text = urllib.parse.unquote(text)
                self.response_queue.put(text)
                if msg['result']['final']:
                    if self.logging: print("Final: {}".format(text))
                    self.response_queue.put("EOS")
                    if self.one_sentence:
                        self.close()
                else:
                    if self.logging: print("Partial: {}".format(text))
            else:
                print("ERROR: {}".format(recvStr))


    def get_full_hyp(self):
        return self.final_hyp_queue.get()

    def closed(self, code, reason=None):
        self.final_hyp_queue.put(" ".join(self.final_hyps))
        self.alive = False

    def is_alive(self):
        return self.alive

    def get_response_queue(self):
        return self.response_queue