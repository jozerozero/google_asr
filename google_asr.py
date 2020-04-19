from __future__ import print_function


import io
import os
import wave
import librosa
import glob
import numpy as np
import pandas as pd
import datetime


from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


if __name__ == '__main__':

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "zijian.json"

    client = speech.SpeechClient()

    # base_path = "/home/zijian/workspace/google_asr/private_key/gupta/gupta_0"
    base_path ='process_wavs'
    # audio_name = "English-NorthAmerica+Female+Child+VoiceBunny_-_ID_4FH32G0_-_Sample_28230-000.wav"

    # audio_path = os.path.join(base_path, audio_name)
    record_file = open(os.path.join(base_path, 'meta.txt'), "wt")

    record_list = list()
    # start_time = datetime.datetime.now()
    for audio_path in os.listdir(base_path):
        if not audio_path.endswith('mp3'):
            continue
        audio_path = os.path.join(base_path, audio_path)
        print(audio_path)
        start_time = datetime.datetime.now()
        with io.open(audio_path, "rb") as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)

        with wave.open(audio_path, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            channel = wave_file.getnchannels()

        y, sample_rate = librosa.core.load(path=audio_path)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=frame_rate,
            audio_channel_count=channel,
            language_code='en-US')

        response = client.recognize(config, audio)
        print(audio_path)
        sentence = ""
        for result in response.results:
            # print('Transcript: {}'.format(result.alternatives[0].transcript))
            sentence += result.alternatives[0].transcript
        record_list.append([audio_path, sentence])
        end_time = datetime.datetime.now()
        print(audio_path+"\t"+sentence+"\n", file=record_file)
        print((end_time - start_time).seconds)

    record = np.asarray(record_list)
    print(record.shape)
    pd.DataFrame(record).to_csv(os.path.join(base_path, "meta.csv"))
    # end_time = datetime.datetime.now()
    # print((end_time - start_time).seconds)

