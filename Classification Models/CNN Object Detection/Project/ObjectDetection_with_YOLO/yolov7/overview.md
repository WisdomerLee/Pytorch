해당 프로젝트 파일은 YOLO git clone을 바탕으로 mask를 쓴 사람, 그렇지 않은 사람 등을 구분하기 위해 사용
yolov7을 이용하고
이미 훈련된 weights를 받아 사용하며 (yolov7-e6e.pt)
cfg/traing/yolov7.yaml파일을 복사하여
yolov7-masks.yaml로 바꾸고
해당 파일에서 nc를 3으로 변경, 

dataset: kaggle.com/datasets/andrewmvd/face-mask-detection
해당 데이터 파일을 사용 - 813개의 데이터
