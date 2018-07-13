import requests
import sys
import pyaudio
import urllib

class TextToSpeech:
    def __init__(self, url='http://10.30.154.10/hmm-stream/syn',
                    key='K9W6tNTeUuwrkyYARkAmzJ94D9vUR2Qdo5YwVI7D',
                    voices='doanngocle.htsvoice'):
        self.url = url
        self.key = key
        self.voices = voices

    def get_speech(self, text):
        payload = {
            'data': text, 
            'voices': self.voices,
            'key': self.key
        }
        print(payload)
        # print(urllib.parse.urlencode(payload))
        r = requests.post(self.url, data=payload)
        return r
    
    def play_audio(self, audio_stream):
        p = pyaudio.PyAudio()
        speaker = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        output=True)
        for chunk in audio_stream.iter_content(chunk_size=1600):
            if chunk:
                speaker.write(chunk)

if __name__ == '__main__':
    tts = TextToSpeech()
    audio = tts.get_speech("Bạn thật là đẹp trai quá đi")
    tts.play_audio(audio)