
import torchaudio
from plot_audio import plot_specgram
import os
import random

# 소리 파일이 담긴 기본 폴더
wav_path = 'data/'
wav_filenames = os.listdir(wav_path)
# 아래는 단순히 얻은 파일 리스트 순서 무작위로 섞기
random.shuffle(wav_filenames)

ALLOWED_CLASSES = ['normal', 'murmur', 'extrahls', 'artifact']

for f in wav_filenames:
  class_type = f.split('_')[0]
  f_index = wave_filenames.index(f)

  target_path = 'train' if f_index < 140 else 'test'
  class_path = f"{target_path}/{class_type}"
  file_path = f"{wav_path}/{f}"
  
