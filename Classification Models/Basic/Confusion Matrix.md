# Confusion Matrix
예측한 구분, 실제 구분, 을 행렬로 만든 것
참, 거짓을 쉽게 구분할 수 있게 함

이진 모델의 경우 예측되는 것도 둘, 실제도 둘이라 2x2의 행렬이 생성
그 중에 둘 다 참인, 둘 다 거짓인 경우만 예측이 성공한 경우고, 그 외의 경우는 에러가 됨

True Pos, True Neg, False Neg, False Pos
True Pos, True Neg가 맞게 예측한 것
False Neg, False Pos가 틀리게 예측한 것

예측과 관련된 Confusion Matrix는
해당 이벤트가 일어날 것으로 예측했는가?
그리고 그 이벤트가 실제로 일어났는가?
의 조합을 놓고 행렬로 만듦
False Alarm - 예측은 했으나 실제 일어나지 않은 것
Miss - 예측도 안 했는데 일어난 것

성능 측정 - 정확성
Numerator - 맞게 예측한 것, Denominator - 가능한 모든 것

Accuracy = Numerator/Denominator
Accuracy = $TP+TN \over TP+TN+FP+FN$

대체로 baseline result, 혹은 모델을 비교하는데 쓸 수 있음

# Performance Measure 성능 측정

Accuracy = $TP+TN \over Total$
measures correct classifier
이 결과를 얼마나 믿을 수 있는가?를 보여주는 측정값

Error Rate = 1 - Accuracy
얼마나 자주 틀리는가?를 보여주는 측정 값

특정 조건일 때 얼마나 맞느냐, 틀리느냐 정도도 confusion matrix로 확인할 수 있음
