from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyaudio
import urllib
import time

from speech_regconize import SpeechRecognize 
from keyword_spotting import KeywordSpotting
from text_to_speech import TextToSpeech
import aiml
    
def main():
    CHUNK = 1024
    CHANNELS = 1
    RATE = 16000
    FORMAT = 'S16LE'
    BYTE_RATE = 32000
    STATUS_KWS = 0
    STATUS_ASR = 1
    STATUS_TTS = 2
    IP = '203.113.152.90'
    IP = '10.30.154.10'

    URI = 'ws://{}:8892/client/ws/speech'.format(IP)
    content_type = "audio/x-raw, layout=(string)interleaved, rate=(int){}, format=(string){}, channels=(int){}".format(RATE, FORMAT, CHANNELS)
    IS_ACTIVE = False  

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
                    
    sr = None
    sentence = None
    kws = KeywordSpotting(graph='graph/conv_kws.pb', labels='graph/conv_kws_labels.txt')
    tts = TextToSpeech(url='http://{}/hmm-stream/syn'.format(IP), voices='doanngocle.htsvoice')
    brain = aiml.Kernel()
    brain.learn("std-startup.xml")
    brain.respond("load aiml b")

    status = STATUS_KWS
    print("===============================")
    print("[INFO] Waiting keyword [xin chào | chào bot | hi bot] ...")
    while True:
        block = stream.read(CHUNK)
        if status == STATUS_KWS:
            if kws.spotting(block):
                status = STATUS_ASR
                print("[INFO] Keyword detected! Start recognize ...")
                sr = SpeechRecognize(url=URI + '?%s' % (urllib.parse.urlencode([("content-type", content_type)])),
                                    byterate=BYTE_RATE,
                                    one_sentence=False)
                sr_response = sr.get_response_queue()
        elif status == STATUS_ASR:
            if sr.is_alive():
                sr.push_audio(block)
                while sr_response.qsize() > 0:
                    msg = sr_response.get()
                    if msg == "EOS":
                        print("\rHuman: {}".format(sentence))
                        text = brain.respond(sentence)
                        audio = tts.get_speech(text)
                        status = STATUS_TTS
                        tts.play_audio(audio)
                        status = STATUS_ASR
                        if "tạm biệt" in sentence:
                            sr.close()
                    else:
                        sentence = msg
                        print("\r-----: {}".format(msg), end='')
                    
            else:
                status = STATUS_KWS
                print("===============================")
                print("========= GOOD BYE!!! =========")
                print("===============================")
                print("[INFO] Waiting keyword ...")
        elif status == STATUS_TTS:
            if sr.is_alive():
                sr.push_audio(bytearray(CHUNK))
            time.sleep(1.*CHUNK/BYTE_RATE)
                


if __name__ == '__main__':
    main()