
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
  # kaggle의 데이터에서 파일 이름이 class이름을 품고 있기 때문에 해당 파일 이름에서 classtype을 추출
  class_type = f.split('_')[0]
  # 파일이 몇 번째인지 확인하고... 
  f_index = wave_filenames.index(f)
  # 만약 파일이 0~139번째 파일이면 훈련 폴더로, 140 이상이면 테스트 폴더로 구분하기
  target_path = 'train' if f_index < 140 else 'test'
  # 아래와 같이 훈련, 테스트 폴더 밑의 클래스 타입으로 구분
  class_path = f"{target_path}/{class_type}"
  # 
  file_path = f"{wav_path}/{f}"
  f_basename = os.path.basename(f)
  f_basename_wo_ext = os.path.splitext(f_basename)[0]
  target_file_path = f"{class_path}/{f_basename_wo_ext}.png"
  if (class_type in ALLOWED_CLASS):
    # 만약 클래스 폴더가 없으면 만들고
    if not os.path.exists(class_path):
      os.makedirs(class_path)
    # load와 함께 waveform으로 변환
    data_waveform, sr = torchaudio.load(file_path)
    # 아래와 같이 file_path를 지정하면 specgram이 지정된 파일 경로에 생성
    plot_specgram(waveform=data_waveform, sample_rate=sr, file_path=target_file_path)
    
