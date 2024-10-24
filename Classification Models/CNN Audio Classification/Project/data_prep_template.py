# 소리 -> 그림으로 변환
import torchaudio
from plot_audio import plot_specgram, plot_waveform
import seaborn as sns
import matplotlib.pyplot as plt

# audio 관련된 것이 설치되었는지 확인할 것
# pip install soundfile
torchaudio.info

wav_file = '.wave' # 저장된 wave file 경로 지정할 것
data_waveform, sr = torchaudio.load(wav_file)

data_waveform.size() # 파일 크기 확인

plot_waveform(data_waveform, sample_rate=sr) # sample rate로 frequency, 소리 파일을 이용하여 waveform의 형태로 그림 그리기 - 해당 부분은 spectrogram을 그릴 때 활용

spectrogram = torchaudio.transforms.Spectrogram()(data_waveform)
spectrogram.size() # 3차원의 tensor

plot_specgram(waveform-data_waveform, sample_rate=sr)
