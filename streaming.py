# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from scipy.io.wavfile import read
import numpy as np
import pyaudio
import struct

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

MODE = 0
FLAGS = None
CHUNK = 1024
CHANNELS = 1
RATE = 16000
SHORT_NORMALIZE = (1.0/32768.0)
RECORD_SECONDS = 1

def read_wav(path):
  sr, data = read(path)
  data = np.array(data, dtype=np.float64)
  if len(data.shape) == 2:
    data = (data[:,0] + data[:,1])/2
  data = data / (1<<15)
  data = np.reshape(data, (len(data), 1))
  return data, sr

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, sample_rate, labels, keyword):
  sample_count = len(wav_data)
  clip_duration_samples = int((FLAGS.clip_duration_ms * sample_rate) / 1000)
  clip_stride_samples = int((FLAGS.clip_stride_ms * sample_rate) / 1000)
  audio_data_end = sample_count - clip_duration_samples
    
  """Runs the audio data through the graph and prints predictions."""

  with tf.Session() as sess:
    if MODE == 0:
      for audio_data_offset in range(0, audio_data_end, clip_stride_samples):
        softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.output_name)
        feed_dicts = {
          FLAGS.input_data_name: wav_data[audio_data_offset:audio_data_offset+clip_duration_samples],
          FLAGS.input_rate_name: sample_rate
          }
        predictions, = sess.run(softmax_tensor, feed_dicts)

        node_id = predictions.argsort()[-1:][::-1][0]
        human_string = labels[node_id]
        score = predictions[node_id]
        if score >= FLAGS.detection_threshold:
          print('%dms: %s (score = %.5f)' % (int(audio_data_offset*1000/sample_rate),human_string, score))
    elif MODE == 1:
      CHUNK = clip_stride_samples
      p = pyaudio.PyAudio()
      stream = p.open(format=pyaudio.paInt16,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)
      wav_data = np.array([])
      audio_data_offset = 0
      while True:
        block = stream.read(CHUNK)
        count = len(block)/2
        format = "%dh"%(count)
        shorts = np.array(list(struct.unpack(format, block)), dtype='float64')
        shorts *= SHORT_NORMALIZE
        wav_data = np.append(wav_data, shorts)
        if(len(wav_data) >= clip_duration_samples):
          wav_data = wav_data[-clip_duration_samples:]
          softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.output_name)
          feed_dicts = {
            FLAGS.input_data_name: np.reshape(wav_data, (len(wav_data), 1)),
            FLAGS.input_rate_name: sample_rate
            }
          predictions, = sess.run(softmax_tensor, feed_dicts)
          
          node_id = predictions.argsort()[-1:][::-1][0]
          human_string = labels[node_id]
          score = predictions[node_id]
          # if human_string != 'silence_' and human_string != '_unknown_':
          print('%dms: %s (score = %.5f)' % (int(audio_data_offset*1000/sample_rate),human_string, score))

          # top_k = predictions.argsort()[-5:][::-1]
          # for node_id in top_k:
          #   human_string = labels[node_id]
          #   score = predictions[node_id]
          #   if(human_string == keyword):
          #     break

          # if score >= FLAGS.detection_threshold:
          #   print('%dms: %s (score = %.5f)' % (int(audio_data_offset*1000/sample_rate),human_string, score))
        
        
        
        audio_data_offset += clip_stride_samples
    return 0


def label_wav(wav, labels, graph, kw):
  global MODE
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    MODE = 1
    print("Streaming mode ...")
    # tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  if MODE == 0:
    # Read wav file
    wav_data, sample_rate = read_wav(wav)
  elif MODE == 1:
    wav_data = []
    sample_rate = RATE
  run_graph(wav_data, sample_rate, labels_list, kw)


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.keyword)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', 
      type=str, 
      default='', 
      help='Audio file to be identified.')
  parser.add_argument(
      '--graph', 
      type=str, 
      default='', 
      help='Model to use for identification.')
  parser.add_argument(
      '--labels', 
      type=str, 
      default='', 
      help='Path to file containing labels.')
  parser.add_argument(
      '--input_data_name',
      type=str,
      default='decoded_sample_data:0',
      help='Name of input data node in model.')
  parser.add_argument(
      '--input_rate_name',
      type=str,
      default='decoded_sample_data:1',
      help='Name of input sample rate node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of output node in model.')
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Length of recognition window.')
  parser.add_argument(
      '--average_window_ms',
      type=int,
      default=500,
      help='Length of window to smooth results over.')
  parser.add_argument(
      '--time_tolerance_ms',
      type=int,
      default=750,
      help='Maximum gap allowed between a recognition and ground truth.')
  parser.add_argument(
      '--suppression_ms',
      type=int,
      default=1500,
      help='How long to ignore others for after a recognition.')
  parser.add_argument(
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition.')
  parser.add_argument(
      '--detection_threshold',
      type=float,
      default=0.6,
      help='What score is required to trigger detection of a word.')
  parser.add_argument(
      '--keyword',
      type=str,
      default='_unknown_',
      help='Keyword to spot.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

#  python3 streaming.py --graph=tmp/my_frozen_graph.pb --labels=tmp/speech_commands_train/conv_labels.txt --detection_threshold=0.8 audio.wav