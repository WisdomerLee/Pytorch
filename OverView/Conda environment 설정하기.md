pytorch.yml
yml 파일이라는 것이 있는데, 
해당 파일에 개발 환경에 필요한 패키지, 버전 등을 모두 작성하여 두기
yml 파일은 개발 환경에 필요한 모든 내용을 담고 있음, 설치된 패키지, 요구되는 것들 등

개발에 쓰이는 코드 편집기를 열어 터미널을 활성화 하고
conda env create -f pytorch.yml
을 입력하기

코드의 interpreter를 pytorch.yml로 설정한 가상환경으로 바꾸어야 함

그리고 yml 파일로 설치가 실패하게 되면 
터미널에 아래와 같이 입력하여 직접 환경을 설정할 것

conda create -n pytorch python
conda activate pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install ipykernel
conda install -c anaconda seaborn
conda install scikit-learn
conda install -c conda-forge detecto
