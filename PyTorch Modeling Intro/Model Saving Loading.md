Deep Learning Model의 훈련에는 몇 시간, 혹은 며칠, 혹은 몇 달이 걸릴 수 있음

그러면 중간에 저장했다가 이어서 하고 싶을 수 있는데
torch.save()라는 함수로 모델의 훈련 내용을 저장할 수 있음
모델 전체를 저장할 필요 없이 모델의 상태만 나타낸 dict 데이터만 저장하는 것을 추천 - 저장, 불러오기 속도 빠름
dictionary는 layers와 연관된 parameters들로 구성되어있음
learnable parameter로 지정된 것들과 batchnorm과 같이 buffer로 지정된 것들만 저장됨

아래와 같이 State dict만 저장하는 방식이 선호됨
통째로 불러오거나 할 수 있으나, 오류 등을 줄이기 위해
모델의 객체를 만들고 state dict만 불러와 덮어쓰는 형태로 접근
```
torch.save(model.state_dict(), PATH)
```

불러오기는 다음과 같이 진행
먼저 모델의 클래스를 이용하여 객체를 만든 뒤
torch.load_state_dict()라는 함수로 학습된 파라미터를 저장
그 뒤엔 모델을 이어서 훈련하거나 평가하거나 진행
model.eval()은 모델을 더 이상 훈련을 진행하지 않고 평가, 활용하는데 사용하는 것

```
model = ModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

또 다른 방식으로 
torch.load()로 모델을 통째로 불러올 수도 있으나... 권장하는 방식은 아님
