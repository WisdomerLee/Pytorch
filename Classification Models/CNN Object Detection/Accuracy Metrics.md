# Object Detection: Accuracy metrics

# Intersection over Union
사물이 그림에서 보이는 상황에서 네모 칸 안에 있음을 예측하는 것 - 사물 분류
그럼 어떻게 예측을 정밀하게 만들 수 있는가?
그 때 쓰이는 기준이 Intersection over Union (IoU)
FP, TP인지 구분하기 - FP, TP는 Confusion matrix의 내용을 참고할 것

IoU = $Intersection \over Total combined area$
두 영역 모두 합친 영역 중에 겹치는 영역의 비율
해당 비율이 높을 수록 정확하게 예측했다고 볼 수 있음

# Derived Metrics
Confusion Matrix를 다시 불러와서
Precision = $TP \over TP+FP$ : 정밀도
Recall = $TP \over TP+FN$ : 물체 감지
F1 score = $Precision*Recall \over (Precision+Recall)/2$ : 

# Precision Recall Curve
좋은 모델은 precision이 높고, recall value가 낮음

모델 여럿을 비교하는 것이라면 Precision Recall Curve를 비교하여 중심 좌표에서 멀 수록 좋은 모델

# Average Precision
precision-recall curve를 하나의 KPI로 요약한 것
해당 값은 0과 1 사이의 값을 가짐
precision-recall curve의 적분 값
AP = $mean(Precision) \over all Recalls$

# Mean Average Precision
평균 Precision을 여러 IoU threshold로 나누기
모든 경우로 평균을 냄
