
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from scipy.io.wavfile import read
import numpy as np
import pyaudio
import struct
from keyword_spotting import KeywordSpotting 

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

MODE = 0
FLAGS = None
CHUNK = 512
CHANNELS = 1
RATE = 16000      
    
def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    ks = KeywordSpotting(graph='graph/conv_kws_350kpositive.pb', labels='graph/conv_kws_350kpositive_labels.txt')
    
    while True:
        block = stream.read(CHUNK)
        results = ks.spotting(block)
        if (results == 1):
            print("Run")
        else:
            print("")





if __name__ == '__main__':
    main()

