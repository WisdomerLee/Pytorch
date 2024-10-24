# 기본
Pytorch는 Audio file에도 동작할 수 있음
데이터를 두 가지 옵션으로 다룰 수 있는데
- 시간 관련으로 다루거나 - 이 방식은 시간이 오래 걸림
- 그림처럼 다루거나 - 이 방식이 더 간편하다고 함
torchaudio라는 패키지에 소리 파일에 실행 가능한 함수들이 들어있음
- audio, signal processing
  - 파일 입, 출력, signal, 데이터 처리 함수
  - datasets
  - model implement
등의 기능을 갖고 있음

# 소리 -> 그림으로 전환하기!
Fast Fourier Transformation
빠른 푸리에 변환을 이용 - 푸리에 변환은 수학적으로 동일한 것을 다르게 표현할 수 있게 도와주는 변환
Time domain으로 있는 소리를 Fourier Transform을 거쳐 Frequency domain의 형태로 바꾸기

Conversion
소리 파일을 작은 시간 단위들로 쪼개어 각각을 파동 기반의 함수로 변환하기
그리고 각각 변환된 것을 하나로 합침

그렇게 합쳐진 것은 Spectrum이라는 형태의 그림 형태로 변환됨
가로축은 시간, 세로축은 파동으로
시간당 주파수를 확인할 수 있음

소리 파일을 스펙트럼이라는 그림의 형태로 변환하고 나면

그 뒤는 이미지 변환과 동일함!
