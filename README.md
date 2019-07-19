# BERT-Activation-Function



## GELU 




## Swish




## Step별 성능 비교
Base Model 기준으로 총 100만 step을 학습을 진행하였고 기존 GELU로 구축한 Model과 성능 비교 결과는 아래와 같습니다. 측정 기준은 step별 KorQuAD Task의 F1,EM으로 측정하였습니다.
<br>

* Base Model(12-layer, 768-hidden, 12-heads)<br>

| Step | seq_length | GELU F1 | GELU EM | swish beta F1 | swish beta EM | swish F1 | swish EM |
|:-------:|:-------:|:-------:| :-------:| :-------:| :-------:| :-------:| :-------:| 
| 40만 | 128 | 77.41% | 62.12% | 00.00% | 00.00% |
| 60만 | 128 |  78.63% | 63.17% | 00.00% | 00.00% |
| 90만 | 128 |  80.98% | 65.25% | 00.00% | 00.00% |
| 100만 | 512 | 91.40% | 79.47% | 00.00% | 00.00% |




## Reference

[Swish, a new activation function for Neural Network](https://jmlb.github.io/ml/2017/12/31/swish_activation_function/)<br>
[Swish in depth: A comparison of Swish & ReLU on CIFAR-10](https://medium.com/@jaiyamsharma/swish-in-depth-a-comparison-of-swish-relu-on-cifar-10-1c798e70ee08)
