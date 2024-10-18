# Receiver Operating Characteristics Curve

원래 2차 세계대전 때 처음 개발되었고, 전투 평원에서 적 오브젝트를 식별하기 위해 개발되었음
이것이 확대되어 모델 성능 측정에도 활용됨

# Confusion Matrix -> ROC curve
Confusion Matrix에서 Threshold의 영향

Threshold를 높이거나, 낮추면 해당 TP, TN, FP, FN의 비율이 변화함

TPR = $TP \over TP+FN$
실제 참인 값들
Y값 ROC curve
FPR = $FP \over FP+TN$
실제 거짓인 값들
X값 ROC curve


# ROC Curve
TPR Y값
FPR X값
평면에서 일직선으로 그어지게 되는 선을 따라가면 완전 무작위로 추정하는 결과와 같고
TPR쪽에 가까우면 나은 분류
FPR쪽에 가까우면 나쁜 분류가 됨

# 목적
서로 다른 메소드를 비교할 수 있게 됨 - ROC Curve로
TPR쪽에 가까워질 수록 더 나은 메소드, 방식!!!

# Model Evaluation: Area Under Curve

Area under curve
ROC의 맵에서 측정된 곳 - 목적은 서로 다른 모델 비교
계산은
곡선의 면적 계산 (적분)
적분

# Model Evaluation: Loss Curve
어느 상황이 더 문제가 생기느냐에 따라 모델을 선택할 수 있음
즉 정확성이 더 중요한가, 혹은 틀리지 않는 것이 중요한가 등
