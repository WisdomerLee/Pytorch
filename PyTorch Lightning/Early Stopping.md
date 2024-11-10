# Early Stopping
모델 훈련에서 선택적인 방법 중에 하나

```
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stop_callback = EarlyStopping(monitor='train_loss', min_delta=0.00, patience=2, verbose=True, mode='min')

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=500, log_every_n_steps=2, callbacks=[early_stop_callback])
trainer.fit(model=model, train_dataloaders=train_loader)
```

대개 훈련 loss 혹은 정확성을 측정하여 훈련의 진행이 더 이상 크게 진전되지 않으면 리소스 낭비 및 과적합을 막기 위해 멈추는 것

