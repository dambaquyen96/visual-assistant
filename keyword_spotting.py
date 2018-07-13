from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import struct
import sys
import os.path
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import numpy as np

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

class KeywordSpotting:

    def __init__(self, wav='', 
                RATE=16000, 
                SHORT_NORMALIZE=(1.0/32768.0), 
                detection_threshold=0.7,
                minimum_length=3,
                clip_duration_ms=1000,
                output_name='labels_softmax:0',
                input_data_name='decoded_sample_data:0',
                input_rate_name='decoded_sample_data:1',
                graph='tmp/my_frozen_graph.pb', 
                labels='tmp/speech_commands_train/conv_labels.txt'):
        self.wav = wav
        self.graph = graph
        self.labels = labels
        
        self.input_data_name = input_data_name
        self.input_rate_name = input_rate_name
        self.output_name = output_name
        self.clip_duration_ms = clip_duration_ms
        self.detection_threshold = detection_threshold
        self.minimum_length = minimum_length
        self.RATE = RATE
        self.SHORT_NORMALIZE = SHORT_NORMALIZE

        self.clip_duration_samples = int((self.clip_duration_ms * self.RATE) / 1000)
        self.wav_data = []
        self.length = 0
        self.last_key = None

        self.labels_list = self.load_labels(self.labels)
        self.load_graph(self.graph)
        

    def load_graph(self, filename):
        """Unpersists graph from file as default graph."""
        with tf.gfile.FastGFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


    def load_labels(self, filename):
        """Read in labels, one label per line."""
        return [line.rstrip() for line in tf.gfile.GFile(filename)]


    def run_graph(self, wav_data, sample_rate, labels):
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name(self.output_name)
            feed_dicts = {
                self.input_data_name: np.reshape(wav_data, (len(wav_data), 1)),
                self.input_rate_name: sample_rate
            }
            predictions, = sess.run(softmax_tensor, feed_dicts)

            node_id = predictions.argsort()[-1:][::-1][0]
            human_string = labels[node_id]
            score = predictions[node_id] 
            return human_string, score


    def spotting(self, block):
        count = len(block)/2
        format = "%dh"%(count)
        shorts = np.array(list(struct.unpack(format, block)), dtype='float64')
        shorts *= self.SHORT_NORMALIZE
        self.wav_data = np.append(self.wav_data, shorts)
        if(len(self.wav_data) >= self.clip_duration_samples):
            self.wav_data = self.wav_data[-self.clip_duration_samples:]
            human_string, score = self.run_graph(self.wav_data, self.RATE, self.labels_list)
            if (human_string == "xinchao" or human_string == "chaobot" or human_string == "haibot") and score >= self.detection_threshold :
                if self.last_key is None:
                    self.last_key = human_string
                    self.length = 1
                else:
                    if self.last_key == human_string:
                        self.length += 1
                    else:
                        self.last_key = human_string
                        self.length = 1
                if self.length >= self.minimum_length:
                    self.last_key = None
                    self.length = 1
                    self.wav_data = []
                    return 1
            else:
                self.last_key = None
                self.length = 1
        return 0
    




